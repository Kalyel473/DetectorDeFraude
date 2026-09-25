"""Contrato comum dos tres modulos de deteccao.

Cada modulo sabe quatro coisas e nada mais:
  1. como carregar seus dados;
  2. como transformar dados em features (fit no treino, transform na predicao);
  3. como identificar um item na saida ("transacao #4471");
  4. como traduzir o nome tecnico de uma feature em portugues de gente.

Isso deixa o pipeline (train/evaluate/predict) totalmente generico.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..features.text_features import TEXT_PREFIX


def fmt_hour(value: float) -> str:
    """3.7833 -> 03:47"""
    value = float(value) % 24
    hours = int(value)
    minutes = int(round((value - hours) * 60))
    if minutes == 60:
        hours, minutes = (hours + 1) % 24, 0
    return f"{hours:02d}:{minutes:02d}"


def fmt_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds >= 86400:
        return f"{seconds / 86400:.1f} dias"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{int(seconds // 60)}min{int(seconds % 60):02d}s"
    return f"{seconds:.0f}s"


def brl(value: float) -> str:
    text = f"{float(value):,.2f}"
    return "R$ " + text.replace(",", "X").replace(".", ",").replace("X", ".")


def text_term_reason(feature: str, value: float) -> str:
    term = feature[len(TEXT_PREFIX):]
    return f"expressao caracteristica no texto: \"{term}\""


def read_table(path) -> pd.DataFrame:
    """Le CSV, JSON (objeto ou lista) ou JSONL para DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suffix in (".json", ".jsonl"):
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        if suffix == ".jsonl" or raw.count("\n{") > 0 and not raw.startswith("["):
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            return pd.DataFrame(records)
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Formato nao suportado: {suffix}")


class FraudModule:
    """Interface implementada pelos tres dominios."""

    key: str = ""
    title: str = ""
    description: str = ""
    sample_file: str = ""
    default_model: str = "rf"
    positive_name: str = "FRAUDE"
    negative_name: str = "LEGITIMO"
    dataset_hint: str = ""

    # ------------------------------------------------------------------ dados
    def load_data(self, path: str | None = None, demo: bool = False) -> pd.DataFrame:
        raise NotImplementedError

    def build_features(self, df: pd.DataFrame, artifacts: dict | None = None):
        """Retorna (X, y, feature_names, artifacts, raw_frame)."""
        raise NotImplementedError

    def texts(self, df: pd.DataFrame) -> list[str]:
        """Texto usado pela camada BERT (vazio nos modulos sem texto)."""
        return []

    def parse_input(self, path: str) -> pd.DataFrame:
        return read_table(path)

    def input_notes(self, df: pd.DataFrame) -> list:
        """Avisos sobre limites da entrada (ex.: features que exigem historico)."""
        return []

    # ------------------------------------------------------------------ saida
    def describe(self, position: int, row: pd.Series) -> str:
        return f"registro #{position}"

    def humanize(self, feature: str, value: float, contribution: float, row: pd.Series) -> str:
        if feature.startswith(TEXT_PREFIX):
            return text_term_reason(feature, value)
        return f"{feature} = {value:.4g}"

    def recommend(self, score: float) -> str:
        if score >= 0.80:
            return "bloquear + verificacao manual"
        if score >= 0.50:
            return "revisar em fila prioritaria"
        if score >= 0.25:
            return "monitorar / pedir segundo fator"
        return "liberar com monitoramento padrao"
