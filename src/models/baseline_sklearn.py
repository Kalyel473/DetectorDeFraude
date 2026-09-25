"""Baselines scikit-learn + tratamento de desbalanceamento (SMOTE).

Regra de ouro anti-leakage: SMOTE roda **somente** no conjunto de treino,
depois do split. Sinteticos no teste inflam a metrica e mentem pro time.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..config import SEED

MODEL_KINDS = ("rf", "gb", "xgb", "logreg", "bert")

MODEL_SHORT = {
    "rf": "RandomForest",
    "gb": "GradientBoosting",
    "xgb": "XGBoost",
    "logreg": "LogisticRegression",
    "bert": "BERT fine-tune",
}

MODEL_LABELS = {
    "rf": "RandomForest (200 arvores, class_weight=balanced)",
    "gb": "GradientBoosting (sklearn)",
    "xgb": "XGBoost (gradient boosting otimizado)",
    "logreg": "LogisticRegression (baseline linear, bom com TF-IDF)",
    "bert": "BERTimbau / DistilBERT fine-tunado (opcional)",
}


def build_estimator(kind: str, seed: int = SEED, n_jobs: int = -1):
    """Instancia o estimador pedido, com fallback avisado quando indisponivel."""
    kind = kind.lower()
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        ), kind
    if kind == "gb":
        return GradientBoostingClassifier(random_state=seed), kind
    if kind == "logreg":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", C=2.0, random_state=seed
        ), kind
    if kind == "xgb":
        try:
            from xgboost import XGBClassifier  # type: ignore

            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=seed,
                n_jobs=n_jobs,
            ), kind
        except Exception:
            return GradientBoostingClassifier(random_state=seed), "gb"
    raise ValueError(f"Modelo desconhecido: {kind}. Use um de {MODEL_KINDS}.")


# --------------------------------------------------------------------------- #
# SMOTE
# --------------------------------------------------------------------------- #
def smote_backend() -> str:
    try:
        import imblearn  # noqa: F401

        return "imbalanced-learn"
    except Exception:
        return "fallback-interno"


def _fallback_smote(X: np.ndarray, y: np.ndarray, seed: int, k: int):
    """SMOTE minimo (interpolacao k-NN) para quando imbalanced-learn nao existe.

    Mesma ideia do original (Chawla et al., 2002): sorteia um vizinho da classe
    minoritaria e cria um ponto no segmento entre os dois.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    major = classes[np.argmax(counts)]
    minor_classes = [c for c in classes if c != major]
    X_out, y_out = [X], [y]
    for cls in minor_classes:
        X_min = X[y == cls]
        need = int(counts.max() - len(X_min))
        if need <= 0 or len(X_min) < 2:
            continue
        k_eff = int(min(k, len(X_min) - 1))
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
        _, idx = nn.kneighbors(X_min)
        base = rng.integers(0, len(X_min), need)
        pick = rng.integers(1, k_eff + 1, need)
        gaps = rng.random((need, 1)).astype(X.dtype)
        neighbours = X_min[idx[base, pick]]
        synthetic = X_min[base] + gaps * (neighbours - X_min[base])
        X_out.append(synthetic.astype(X.dtype))
        y_out.append(np.full(need, cls, dtype=y.dtype))
    return np.vstack(X_out), np.concatenate(y_out)


def apply_smote(X: np.ndarray, y: np.ndarray, seed: int = SEED, enabled: bool = True) -> tuple:
    """Retorna (X_res, y_res, info). info descreve o que realmente aconteceu."""
    counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    info = {
        "applied": False,
        "backend": None,
        "before": counts,
        "after": counts,
        "reason": None,
    }
    if not enabled:
        info["reason"] = "desativado por --no-smote"
        return X, y, info
    if len(counts) < 2:
        info["reason"] = "apenas uma classe presente no treino"
        return X, y, info

    minority = min(counts.values())
    if minority < 6:
        info["reason"] = f"classe minoritaria pequena demais ({minority} amostras)"
        return X, y, info

    k = int(min(5, minority - 1))
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore

        X_res, y_res = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
        info["backend"] = "imbalanced-learn"
    except Exception:
        X_res, y_res = _fallback_smote(X, y, seed, k)
        info["backend"] = "fallback-interno"

    info["applied"] = True
    info["after"] = {int(c): int(n) for c, n in zip(*np.unique(y_res, return_counts=True))}
    return X_res, y_res, info


def fit_model(kind: str, X: np.ndarray, y: np.ndarray, seed: int = SEED, use_smote: bool = True):
    """Aplica SMOTE no treino e ajusta o estimador. Retorna (model, kind_real, info)."""
    estimator, real_kind = build_estimator(kind, seed=seed)
    X_res, y_res, smote_info = apply_smote(X, y, seed=seed, enabled=use_smote)
    estimator.fit(X_res, y_res)
    return estimator, real_kind, smote_info


def positive_proba(model, X: np.ndarray) -> np.ndarray:
    """Probabilidade da classe 1 (fraude), com fallback para decision_function."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            classes = list(getattr(model, "classes_", [0, 1]))
            index = classes.index(1) if 1 in classes else proba.shape[1] - 1
            return proba[:, index]
        return proba.ravel()
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))
