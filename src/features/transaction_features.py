"""Engenharia de features para fraude em transacoes de cartao.

Compatibilidade de dataset:
  * Credit Card Fraud Detection (ULB/Kaggle): Time, V1..V28, Amount, Class.
    As colunas V1..V28 sao componentes PCA - dados sensiveis mascarados,
    exatamente a pratica usada em compliance (PCI-DSS / LGPD).
  * IEEE-CIS ou extratos proprios: qualquer CSV com colunas de valor e tempo.
  * Se existir uma coluna de cartao (card_id / card / CardID), as features de
    velocidade sao calculadas por cartao. Sem ela, a serie e tratada como
    um fluxo unico (degradacao controlada, nunca quebra).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TIME_CANDIDATES = ("Time", "time", "timestamp", "TransactionDT")
AMOUNT_CANDIDATES = ("Amount", "amount", "valor", "TransactionAmt")
CARD_CANDIDATES = ("card_id", "cardId", "CardID", "card", "cartao")
LABEL_CANDIDATES = ("Class", "class", "label", "is_fraud", "isFraud", "fraude")

NIGHT_HOURS = (0, 6)          # madrugada: janela classica de teste de cartao
BURST_WINDOW = 3600.0         # 1 hora, para contagem de transacoes recentes
BURST_TIGHT = 120.0           # 2 minutos, para o flag de velocidade

ENGINEERED_COLS = [
    "amount",
    "amount_log",
    "hour_of_day",
    "is_night",
    "secs_since_last_tx",
    "tx_count_last_hour",
    "amount_ratio_card_mean",
    "amount_zscore_global",
    "velocity_flag",
]


def _pick(df: pd.DataFrame, candidates) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def find_label_col(df: pd.DataFrame) -> str | None:
    return _pick(df, LABEL_CANDIDATES)


def pca_columns(df: pd.DataFrame) -> list[str]:
    """V1..V28 do dataset da ULB (ou qualquer V<numero>)."""
    cols = [c for c in df.columns if len(c) > 1 and c[0] == "V" and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def _velocity_by_group(times: np.ndarray, amounts: np.ndarray):
    """Retorna (gap_desde_ultima, qtd_na_ultima_hora, razao_vs_media_anterior).

    Calculado apenas com o passado de cada grupo - sem olhar o futuro, para
    nao criar data leakage temporal.
    """
    order = np.argsort(times, kind="stable")
    t = times[order].astype(float)
    a = amounts[order].astype(float)

    gap = np.diff(t, prepend=t[0] - 7 * 86400.0)
    left = np.searchsorted(t, t - BURST_WINDOW, side="left")
    count = np.arange(len(t)) - left + 1

    running_mean = np.cumsum(a) / np.arange(1, len(a) + 1)
    prev_mean = np.concatenate([[np.nan], running_mean[:-1]])
    ratio = np.where(np.isnan(prev_mean), 1.0, a / np.maximum(prev_mean, 0.01))

    out_gap = np.empty_like(gap)
    out_count = np.empty_like(count, dtype=float)
    out_ratio = np.empty_like(ratio)
    out_gap[order] = gap
    out_count[order] = count
    out_ratio[order] = ratio
    return out_gap, out_count, out_ratio


def engineer(df: pd.DataFrame, stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Deriva as features comportamentais. `stats` guarda media/desvio do treino."""
    df = df.reset_index(drop=True)
    time_col = _pick(df, TIME_CANDIDATES)
    amount_col = _pick(df, AMOUNT_CANDIDATES)
    card_col = _pick(df, CARD_CANDIDATES)

    n = len(df)
    amount = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0).to_numpy(dtype=float) \
        if amount_col else np.zeros(n)
    time = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0).to_numpy(dtype=float) \
        if time_col else np.arange(n, dtype=float) * 60.0

    out = pd.DataFrame(index=df.index)
    out["amount"] = amount
    out["amount_log"] = np.log1p(np.clip(amount, 0, None))

    # Time no dataset da ULB = segundos desde a primeira transacao do dataset.
    hour = (time / 3600.0) % 24.0
    out["hour_of_day"] = hour
    out["is_night"] = ((hour >= NIGHT_HOURS[0]) & (hour < NIGHT_HOURS[1])).astype(int)

    gap = np.full(n, 7 * 86400.0)
    count = np.ones(n)
    ratio = np.ones(n)
    if card_col is not None:
        for _, group in df.groupby(card_col, sort=False):
            pos = group.index.to_numpy()
            g, c, r = _velocity_by_group(time[pos], amount[pos])
            gap[pos], count[pos], ratio[pos] = g, c, r
    elif n > 1:
        gap, count, ratio = _velocity_by_group(time, amount)

    out["secs_since_last_tx"] = gap
    out["tx_count_last_hour"] = count
    out["amount_ratio_card_mean"] = np.clip(ratio, 0, 500)

    stats = dict(stats or {})
    if "amount_mean" not in stats:
        stats["amount_mean"] = float(np.mean(amount)) if n else 0.0
        stats["amount_std"] = float(np.std(amount)) or 1.0
        stats["has_card_col"] = card_col is not None
    out["amount_zscore_global"] = (amount - stats["amount_mean"]) / max(stats["amount_std"], 1e-9)
    out["velocity_flag"] = ((count >= 3) & (gap <= BURST_TIGHT)).astype(int)
    return out[ENGINEERED_COLS], stats


def build_matrix(df: pd.DataFrame, artifacts: dict | None = None):
    """(X, y, feature_names, artifacts). artifacts=None => modo fit (treino)."""
    df = df.reset_index(drop=True)
    fitting = artifacts is None
    artifacts = dict(artifacts or {})

    engineered, stats = engineer(df, artifacts.get("stats"))
    artifacts["stats"] = stats

    if fitting:
        pca_cols = pca_columns(df)
    else:
        pca_cols = artifacts.get("pca_cols", [])
        for col in pca_cols:                     # predicao com colunas faltando
            if col not in df.columns:
                df[col] = 0.0
    artifacts["pca_cols"] = pca_cols

    frame = pd.concat([df[pca_cols].astype(float).fillna(0.0), engineered], axis=1)
    feature_names = list(frame.columns)
    values = frame.to_numpy(dtype=np.float32)

    if fitting:
        scaler = StandardScaler().fit(values)
        artifacts["scaler"] = scaler
    scaler = artifacts["scaler"]
    X = scaler.transform(values).astype(np.float32)
    artifacts["feature_names"] = feature_names
    artifacts["raw_frame_cols"] = feature_names

    label_col = find_label_col(df)
    y = df[label_col].astype(int).to_numpy() if label_col else None
    artifacts["label_col"] = label_col
    return X, y, feature_names, artifacts, frame
