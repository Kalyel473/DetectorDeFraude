"""Caminhos, constantes e limiares globais do FraudShield BR."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SEED = 42
TEST_SIZE = 0.25

# Limiares de risco usados na saida do terminal e nas recomendacoes.
RISK_BANDS = (
    (0.80, "ALTO RISCO"),
    (0.50, "RISCO MEDIO"),
    (0.25, "RISCO BAIXO"),
    (0.00, "SEM INDICIO"),
)

# Quantas features aparecem na explicacao individual de cada predicao.
TOP_FACTORS = 5


def risk_label(score: float) -> str:
    """Traduz o score continuo (0-1) para a faixa de risco textual."""
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label
    return "SEM INDICIO"


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, SAMPLES_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def model_path(module: str, kind: str) -> Path:
    """models/transaction_rf.joblib, models/email_bert/, ..."""
    if kind == "bert":
        return MODELS_DIR / f"{module}_bert"
    return MODELS_DIR / f"{module}_{kind}.joblib"


def report_dir(module: str) -> Path:
    path = REPORTS_DIR / module
    path.mkdir(parents=True, exist_ok=True)
    return path
