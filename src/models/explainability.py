"""Explicabilidade: por que o modelo chamou isto de fraude.

Caminho principal: SHAP (TreeExplainer para floresta/boosting, LinearExplainer
para modelos lineares). Caminho de fallback, quando `shap` nao esta instalado:
explicacao por OCLUSAO - troca-se uma feature pelo valor tipico do treino e
mede-se quanto o score cai. Menos rigoroso que Shapley, mas honesto, rapido e
sempre disponivel. A saida sempre informa qual metodo foi usado.
"""
from __future__ import annotations

import warnings

import numpy as np

try:
    import shap  # type: ignore

    HAS_SHAP = True
except Exception:  # pragma: no cover
    shap = None
    HAS_SHAP = False

TREE_MODELS = ("RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier",
               "ExtraTreesClassifier", "DecisionTreeClassifier")


def method_name() -> str:
    return "SHAP" if HAS_SHAP else "oclusao (fallback interno)"


def _is_tree(model) -> bool:
    return type(model).__name__ in TREE_MODELS


def _positive_class_index(model) -> int:
    classes = list(getattr(model, "classes_", [0, 1]))
    return classes.index(1) if 1 in classes else len(classes) - 1


def _normalize_shap(values, model) -> np.ndarray:
    """Reduz qualquer formato de retorno do shap para (n_amostras, n_features)."""
    idx = _positive_class_index(model)
    if isinstance(values, list):
        values = values[min(idx, len(values) - 1)]
    values = np.asarray(values)
    if values.ndim == 3:                       # (n, features, classes)
        values = values[:, :, min(idx, values.shape[2] - 1)]
    return np.asarray(values, dtype=float)


def shap_values(model, X: np.ndarray, background: np.ndarray | None = None) -> np.ndarray | None:
    """Valores SHAP (n, f) para a classe positiva, ou None se indisponivel."""
    if not HAS_SHAP:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if _is_tree(model):
                explainer = shap.TreeExplainer(model)
                return _normalize_shap(explainer.shap_values(X, check_additivity=False), model)
            if hasattr(model, "coef_"):
                ref = background if background is not None else X
                explainer = shap.LinearExplainer(model, ref)
                return _normalize_shap(explainer.shap_values(X), model)
            ref = background if background is not None else X[:50]
            explainer = shap.KernelExplainer(lambda d: model.predict_proba(d)[:, 1], ref[:50])
            return _normalize_shap(explainer.shap_values(X, nsamples=100, silent=True), model)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Importancia global
# --------------------------------------------------------------------------- #
def global_importance(
    model,
    X: np.ndarray,
    feature_names: list[str],
    top_k: int = 15,
    max_samples: int = 300,
) -> tuple[list[tuple[str, float]], str]:
    """Top features do modelo como um todo. Retorna (lista, metodo)."""
    sample = X[:max_samples]
    values = shap_values(model, sample)
    if values is not None and values.size:
        scores = np.abs(values).mean(axis=0)
        method = "SHAP (media do |valor| por feature)"
    elif hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_, dtype=float)
        method = "importancia por impureza (Gini)"
    elif hasattr(model, "coef_"):
        scores = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        method = "magnitude dos coeficientes"
    else:
        return [], "indisponivel"

    scores = scores[: len(feature_names)]
    order = np.argsort(scores)[::-1][:top_k]
    return [(feature_names[i], float(scores[i])) for i in order], method


def plot_global_importance(pairs, path, title: str) -> str | None:
    """Grafico de barras horizontais com as features mais influentes."""
    if not pairs:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [p[0] for p in pairs][::-1]
        values = [p[1] for p in pairs][::-1]
        height = max(3.0, 0.38 * len(names))
        fig, ax = plt.subplots(figsize=(9, height))
        ax.barh(names, values, color="#1f9d55")
        ax.set_title(title)
        ax.set_xlabel("influencia media na decisao")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Explicacao individual
# --------------------------------------------------------------------------- #
def _occlusion_contributions(model, x: np.ndarray, baseline: np.ndarray, candidates: np.ndarray):
    """Queda no score ao substituir cada feature candidata pelo valor tipico."""
    from .baseline_sklearn import positive_proba

    base_score = float(positive_proba(model, x.reshape(1, -1))[0])
    if len(candidates) == 0:
        return np.array([]), base_score
    batch = np.repeat(x.reshape(1, -1), len(candidates), axis=0)
    for row, idx in enumerate(candidates):
        batch[row, idx] = baseline[idx]
    scores = positive_proba(model, batch)
    return base_score - scores, base_score


def instance_contributions(
    model,
    x: np.ndarray,
    feature_names: list[str],
    background: np.ndarray | None = None,
    top_k: int = 5,
    max_candidates: int = 60,
) -> tuple[list[tuple[str, float]], str]:
    """Top features que empurraram ESTA amostra para fraude. (nome, contribuicao)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    values = shap_values(model, x.reshape(1, -1), background)
    if values is not None and values.size:
        contrib = values[0][: len(feature_names)]
        method = "SHAP"
    else:
        if background is None or not len(background):
            return [], "indisponivel"
        baseline = np.median(background, axis=0)
        if hasattr(model, "feature_importances_"):
            ranking = np.argsort(np.asarray(model.feature_importances_))[::-1]
        elif hasattr(model, "coef_"):
            ranking = np.argsort(np.abs(np.asarray(model.coef_).ravel()))[::-1]
        else:
            ranking = np.arange(len(feature_names))
        candidates = np.array([i for i in ranking[:max_candidates] if i < len(feature_names)])
        deltas, _ = _occlusion_contributions(model, x, baseline, candidates)
        contrib = np.zeros(len(feature_names))
        contrib[candidates] = deltas
        method = "oclusao"

    order = np.argsort(np.abs(contrib))[::-1]
    picked = [(feature_names[i], float(contrib[i])) for i in order if contrib[i] > 0][:top_k]
    if not picked:                                   # nada empurrou para fraude
        picked = [(feature_names[i], float(contrib[i])) for i in order[:top_k]]
    return picked, method
