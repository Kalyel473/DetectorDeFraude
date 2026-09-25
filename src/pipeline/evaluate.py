"""Avaliacao: as metricas que importam quando fraude e 0,17% da base.

Por que accuracy sozinha engana: num dataset com 0,172% de fraude, o modelo
preguicoso que responde "legitimo" para tudo acerta 99,828%. Precision, Recall,
F1 e principalmente a curva Precision-Recall sao o que mostram se o modelo
realmente encontra fraude - e quanto de falso alarme isso custa pro time.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .. import ui


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    positives = int(y_true.sum())
    majority = float(max(1 - y_true.mean(), y_true.mean())) if len(y_true) else 0.0
    two_classes = len(np.unique(y_true)) > 1

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positives": positives,
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "majority_accuracy": majority,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if two_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if two_classes else None,
        "confusion": matrix.tolist(),
        "report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["legitimo", "fraude"], zero_division=0
        ),
    }
    if two_classes:
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1s = np.divide(
            2 * precision * recall, precision + recall,
            out=np.zeros_like(precision), where=(precision + recall) > 0,
        )
        best = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
        metrics["best_threshold"] = float(thresholds[min(best, len(thresholds) - 1)])
        metrics["best_f1"] = float(f1s[best])
    return metrics


# --------------------------------------------------------------------------- #
# Graficos
# --------------------------------------------------------------------------- #
def plot_curves(y_true, y_score, out_dir: Path, prefix: str = "") -> dict:
    """ROC, Precision-Recall e matriz de confusao em PNG. Retorna os caminhos."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    green = "#1f9d55"

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=green, lw=2, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="aleatorio")
    ax.set_xlabel("taxa de falso positivo")
    ax.set_ylabel("taxa de verdadeiro positivo")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    paths["roc"] = str(out_dir / f"{prefix}roc.png")
    fig.savefig(paths["roc"], dpi=130)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=green, lw=2, label=f"PR-AUC = {ap:.4f}")
    ax.axhline(y_true.mean(), ls="--", color="grey", lw=1,
               label=f"base (prevalencia = {y_true.mean():.4f})")
    ax.set_xlabel("recall (fraudes encontradas)")
    ax.set_ylabel("precision (alarmes corretos)")
    ax.set_title("Curva Precision-Recall - mais informativa em classe rara")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    paths["pr"] = str(out_dir / f"{prefix}precision_recall.png")
    fig.savefig(paths["pr"], dpi=130)
    plt.close(fig)

    matrix = confusion_matrix(y_true, (y_score >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(matrix, cmap="Greens")
    ax.set_xticks([0, 1], ["previsto legitimo", "previsto fraude"])
    ax.set_yticks([0, 1], ["real legitimo", "real fraude"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center",
                    color="black", fontsize=15, fontweight="bold")
    ax.set_title("Matriz de confusao")
    fig.tight_layout()
    paths["confusion"] = str(out_dir / f"{prefix}matriz_confusao.png")
    fig.savefig(paths["confusion"], dpi=130)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# Saida no terminal
# --------------------------------------------------------------------------- #
def print_metrics(metrics: dict, title: str = "Metricas no conjunto de teste") -> None:
    ui.section(title)
    rows = [
        ["Accuracy", f"{metrics['accuracy']:.4f}",
         f"chute na classe majoritaria daria {metrics['majority_accuracy']:.4f} - por isso ela engana"],
        ["Precision", f"{metrics['precision']:.4f}", "dos alarmes disparados, quantos eram fraude"],
        ["Recall", f"{metrics['recall']:.4f}", "das fraudes existentes, quantas foram pegas"],
        ["F1-score", f"{metrics['f1']:.4f}", "equilibrio entre os dois acima"],
        ["ROC-AUC", "-" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}",
         "separacao geral entre as classes"],
        ["PR-AUC", "-" if metrics["pr_auc"] is None else f"{metrics['pr_auc']:.4f}",
         "metrica principal em classe rara"],
    ]
    if "best_threshold" in metrics:
        rows.append(["Melhor limiar", f"{metrics['best_threshold']:.3f}",
                     f"F1 maximo de {metrics['best_f1']:.4f} ajustando o corte"])
    ui.table(["metrica", "valor", "leitura"], rows)
    ui.raw()
    ui.confusion(metrics["confusion"])
    tn, fp = metrics["confusion"][0]
    fn, tp = metrics["confusion"][1]
    ui.raw()
    ui.bullets([
        f"{tp} fraudes detectadas e {fn} perdidas (falsos negativos = prejuizo direto)",
        f"{fp} falsos alarmes sobre {tn + fp} legitimas (custo operacional da fila de revisao)",
    ])


def comparison_table(rows: list[dict], title: str = "Comparativo de modelos") -> None:
    """rows: [{'modelo','precision','recall','f1','pr_auc','roc_auc','treino_s','custo'}]"""
    ui.section(title)
    body = []
    for row in rows:
        body.append([
            row.get("modelo", "?"),
            f"{row.get('precision', 0):.4f}",
            f"{row.get('recall', 0):.4f}",
            f"{row.get('f1', 0):.4f}",
            "-" if row.get("pr_auc") is None else f"{row.get('pr_auc'):.4f}",
            "-" if row.get("roc_auc") is None else f"{row.get('roc_auc'):.4f}",
            f"{row.get('treino_s', 0):.1f}s",
            row.get("custo", "-"),
        ])
    ui.table(
        ["modelo", "precision", "recall", "F1", "PR-AUC", "ROC-AUC", "treino", "custo"],
        body,
    )


# --------------------------------------------------------------------------- #
# Relatorio em disco
# --------------------------------------------------------------------------- #
def save_report(metrics: dict, out_dir: Path, module: str, model_label: str,
                importance: list | None = None, extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "modulo": module,
        "modelo": model_label,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "metricas": {k: v for k, v in metrics.items() if k != "report"},
        "top_features": [{"feature": f, "peso": round(v, 6)} for f, v in (importance or [])],
    }
    if extra:
        payload.update(extra)
    json_path = out_dir / "metricas.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# FraudShield BR - relatorio do modulo `{module}`",
        "",
        f"- Modelo: **{model_label}**",
        f"- Gerado em: {payload['gerado_em']}",
        f"- Amostras de teste: {metrics['n']} ({metrics['positives']} fraudes, "
        f"{metrics['positive_rate'] * 100:.2f}%)",
        "",
        "## Metricas",
        "",
        "| metrica | valor |",
        "| --- | --- |",
        f"| accuracy | {metrics['accuracy']:.4f} |",
        f"| accuracy do chute majoritario | {metrics['majority_accuracy']:.4f} |",
        f"| precision | {metrics['precision']:.4f} |",
        f"| recall | {metrics['recall']:.4f} |",
        f"| f1 | {metrics['f1']:.4f} |",
        f"| roc_auc | {metrics['roc_auc'] if metrics['roc_auc'] is None else round(metrics['roc_auc'], 4)} |",
        f"| pr_auc | {metrics['pr_auc'] if metrics['pr_auc'] is None else round(metrics['pr_auc'], 4)} |",
        "",
        "## Matriz de confusao",
        "",
        "|  | previsto legitimo | previsto fraude |",
        "| --- | --- | --- |",
        f"| real legitimo | {metrics['confusion'][0][0]} | {metrics['confusion'][0][1]} |",
        f"| real fraude | {metrics['confusion'][1][0]} | {metrics['confusion'][1][1]} |",
        "",
        "## Classification report",
        "",
        "```",
        metrics.get("report", ""),
        "```",
    ]
    if importance:
        lines += ["", "## Features mais influentes", ""]
        lines += [f"{i + 1}. `{f}` - {v:.6f}" for i, (f, v) in enumerate(importance)]
    md_path = out_dir / "relatorio.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
