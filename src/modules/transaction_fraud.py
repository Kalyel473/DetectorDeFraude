"""Modulo 1 - fraude em transacoes de cartao.

Dataset de referencia: Credit Card Fraud Detection (ULB / Kaggle) -
284.807 transacoes, 492 fraudes (0,172%). O extremo desbalanceamento e o
melhor argumento didatico contra usar accuracy como metrica unica: um modelo
que chuta "tudo legitimo" acerta 99,83% e nao detecta nada.

Alternativa com mais features categoricas: IEEE-CIS Fraud Detection.
Coloque o CSV em data/raw/ e rode com --data data/raw/creditcard.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SAMPLES_DIR, RAW_DIR
from ..features import transaction_features as tfeat
from ..features.text_features import TEXT_PREFIX
from .base import FraudModule, brl, fmt_duration, fmt_hour, read_table

SAMPLE_NAME = "transacoes_amostra.csv"
RAW_CANDIDATES = ("creditcard.csv", "transacoes.csv", "train_transaction.csv")


class TransactionFraudModule(FraudModule):
    key = "transaction"
    title = "Fraude em transacoes de cartao"
    description = "Valor, velocidade, horario e componentes PCA anonimizados (V1-V28)"
    sample_file = SAMPLE_NAME
    default_model = "rf"
    positive_name = "FRAUDE"
    negative_name = "LEGITIMA"
    dataset_hint = (
        "Baixe o Credit Card Fraud Detection (ULB/Kaggle) como data/raw/creditcard.csv "
        "ou use --demo para rodar com a amostra do repositorio."
    )

    # ------------------------------------------------------------------ dados
    def load_data(self, path: str | None = None, demo: bool = False) -> pd.DataFrame:
        if path:
            return read_table(path)
        if not demo:
            for name in RAW_CANDIDATES:
                candidate = RAW_DIR / name
                if candidate.exists():
                    return read_table(candidate)
        return read_table(SAMPLES_DIR / SAMPLE_NAME)

    def build_features(self, df: pd.DataFrame, artifacts: dict | None = None):
        return tfeat.build_matrix(df, artifacts)

    def input_notes(self, df: pd.DataFrame) -> list:
        notes = []
        if len(df) < 30:
            notes.append(
                "as features de velocidade (gap desde a ultima transacao, contagem na ultima "
                "hora, razao vs media do cartao) precisam do historico: com poucas linhas elas "
                "saem neutras e o score fica conservador. Passe o extrato completo do cartao "
                "para a analise usar o pipeline inteiro."
            )
        if not any(col in df.columns for col in tfeat.CARD_CANDIDATES):
            notes.append(
                "sem coluna de cartao (card_id): a velocidade e calculada sobre o fluxo todo, "
                "nao por cartao."
            )
        return notes

    # ------------------------------------------------------------------ saida
    def describe(self, position: int, row: pd.Series) -> str:
        for col in ("tx_id", "TransactionID", "id"):
            if col in row.index and pd.notna(row[col]):
                return f"transacao #{row[col]}"
        return f"transacao #{position}"

    def humanize(self, feature: str, value: float, contribution: float, row: pd.Series) -> str:
        value = float(value)
        if feature.startswith(TEXT_PREFIX):
            return f"termo suspeito: {feature[len(TEXT_PREFIX):]}"

        if feature == "amount_ratio_card_mean":
            if value >= 1.2:
                return f"valor {value:.1f}x acima da media historica do cartao (+{(value - 1) * 100:.0f}%)"
            return f"valor {value:.2f}x da media historica do cartao"
        if feature == "amount":
            return f"valor da transacao: {brl(value)}"
        if feature == "amount_log":
            # a feature e log1p(valor); volta para reais para a frase fazer sentido
            return f"valor da transacao: {brl(float(np.expm1(value)))} (escala log)"
        if feature == "amount_zscore_global":
            return f"valor a {value:.1f} desvios-padrao da media da carteira"
        if feature == "hour_of_day":
            return f"horario incomum ({fmt_hour(value)})"
        if feature == "is_night":
            return "transacao na madrugada (00h-06h)" if value else "transacao em horario comercial"
        if feature == "secs_since_last_tx":
            return f"apenas {fmt_duration(value)} desde a transacao anterior do cartao"
        if feature == "tx_count_last_hour":
            return f"velocidade de transacao ({value:.0f} transacoes na ultima hora)"
        if feature == "velocity_flag":
            return "rajada detectada (3+ transacoes em 2 minutos)" if value else "sem rajada de transacoes"
        if feature.startswith("V") and feature[1:].isdigit():
            return f"componente PCA {feature} = {value:.2f} (padrao anonimizado associado a fraude)"
        return f"{feature} = {value:.4g}"
