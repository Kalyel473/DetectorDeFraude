"""Treino: dados brutos -> features -> split -> SMOTE -> fit -> metricas -> SHAP.

O split acontece ANTES do SMOTE. Isso nao e detalhe de estilo: sintetizar
minoria antes de separar treino e teste vaza informacao e infla a metrica.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from .. import ui
from ..config import SEED, TEST_SIZE, model_path, report_dir
from ..models import explainability as xai
from ..models.baseline_sklearn import MODEL_LABELS, fit_model, positive_proba
from ..modules.base import FraudModule
from . import evaluate

BACKGROUND_SIZE = 300      # amostra guardada no modelo para explicar predicoes


def _split(n: int, y: np.ndarray, test_size: float, seed: int):
    stratify = y if len(np.unique(y)) > 1 else None
    return train_test_split(
        np.arange(n), test_size=test_size, random_state=seed, stratify=stratify
    )


def run_train(
    module: FraudModule,
    data_path: str | None = None,
    demo: bool = False,
    model_kind: str | None = None,
    use_smote: bool = True,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
    save: bool = True,
    make_plots: bool = True,
    explain: bool = True,
    top_k: int = 15,
    bert_epochs: int = 2,
    bert_batch: int = 16,
    bert_model: str | None = None,
) -> dict:
    kind = (model_kind or module.default_model).lower()

    # ------------------------------------------------------------------ dados
    ui.section(f"Modulo {module.key} - {module.title}")
    ui.step("Carregando dados...")
    frame = module.load_data(data_path, demo=demo)
    label_col_hint = "label"
    ui.ok(f"{len(frame):,} registros carregados".replace(",", "."))

    # --------------------------------------------------- engenharia de features
    ui.step("Engenharia de features...")
    started = time.perf_counter()
    X, y, feature_names, artifacts, raw = module.build_features(frame, None)
    if y is None:
        raise SystemExit(
            f"Dataset sem coluna de rotulo. O treino precisa de uma coluna de classe "
            f"(ex.: Class/label/is_fraud). {module.dataset_hint}"
        )
    label_col = artifacts.get("label_col", label_col_hint)
    positives = int(np.sum(y))
    ui.ok(f"{X.shape[1]} features construidas a partir de {X.shape[0]} registros")
    ui.kv([
        ("coluna de rotulo", label_col),
        ("fraudes na base", f"{positives} ({positives / max(len(y), 1) * 100:.3f}%)"),
        ("features de texto", sum(1 for f in feature_names if f.startswith("txt:"))),
        ("tempo de extracao", f"{time.perf_counter() - started:.2f}s"),
    ])

    # ------------------------------------------------------------------ split
    ui.step(f"Split treino/teste ({int((1 - test_size) * 100)}/{int(test_size * 100)}, estratificado)...")
    idx_train, idx_test = _split(len(y), y, test_size, seed)
    y_train, y_test = y[idx_train], y[idx_test]
    ui.ok(f"treino: {len(idx_train)} ({int(y_train.sum())} fraudes) | "
          f"teste: {len(idx_test)} ({int(y_test.sum())} fraudes)")

    bundle: dict = {
        "module": module.key,
        "model_kind": kind,
        "feature_names": feature_names,
        "artifacts": artifacts,
        "threshold": 0.5,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(data_path or ("amostra --demo" if demo else "data/raw")),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
    }

    # ------------------------------------------------------------------ treino
    importance: list = []
    importance_method = "-"
    if kind == "bert":
        from ..models.bert_classifier import DEFAULT_MODEL, BertTextClassifier, availability

        ok, message = availability()
        ui.step(f"Camada BERT: {message}")
        if not ok:
            raise SystemExit(
                "Para --model bert instale: pip install transformers torch\n"
                "Alternativa sem GPU/downloads: --model rf (baseline sklearn)."
            )
        texts = module.texts(frame)
        if not texts:
            raise SystemExit(f"O modulo {module.key} nao expoe texto para fine-tuning de BERT.")
        clf = BertTextClassifier(
            model_name=bert_model or DEFAULT_MODEL,
            epochs=bert_epochs,
            batch_size=bert_batch,
            seed=seed,
        )
        ui.step(f"Fine-tuning de {clf.model_name} ({bert_epochs} epocas)...")
        started = time.perf_counter()
        clf.fit([texts[i] for i in idx_train], y_train)
        train_seconds = time.perf_counter() - started
        scores = clf.predict_proba([texts[i] for i in idx_test])[:, 1]
        smote_info = {"applied": False, "reason": "texto nao usa SMOTE (peso de classe na perda)"}
        model_label = f"BERT fine-tunado ({clf.model_name})"
        bundle["bert_dir"] = str(model_path(module.key, "bert"))
        trained_object = clf
    else:
        X_train, X_test = X[idx_train], X[idx_test]
        ui.step("Balanceamento com SMOTE (somente no treino)...")
        started = time.perf_counter()
        model, real_kind, smote_info = fit_model(kind, X_train, y_train, seed=seed, use_smote=use_smote)
        train_seconds = time.perf_counter() - started
        if smote_info["applied"]:
            ui.ok(f"SMOTE aplicado ({smote_info['backend']}): "
                  f"{smote_info['before']} -> {smote_info['after']}")
        else:
            ui.warn(f"SMOTE nao aplicado: {smote_info['reason']}")
        if real_kind != kind:
            ui.warn(f"modelo '{kind}' indisponivel, usando '{real_kind}' no lugar")
        kind = real_kind
        bundle["model_kind"] = kind
        model_label = MODEL_LABELS.get(kind, kind)
        ui.ok(f"{model_label} treinado em {train_seconds:.2f}s")
        scores = positive_proba(model, X_test)
        trained_object = model
        bundle["model"] = model
        background = X_train[np.random.default_rng(seed).permutation(len(X_train))[:BACKGROUND_SIZE]]
        bundle["background"] = background

    # --------------------------------------------------------------- metricas
    metrics = evaluate.compute_metrics(y_test, scores, threshold=0.5)
    evaluate.print_metrics(metrics, f"Metricas no teste - {model_label}")

    plots = {}
    if make_plots:
        plots = evaluate.plot_curves(y_test, scores, report_dir(module.key), prefix=f"{kind}_")
        if plots:
            ui.raw()
            ui.info("graficos salvos: " + ", ".join(Path(p).name for p in plots.values()))

    # ---------------------------------------------------------- explicabilidade
    if explain and kind != "bert":
        ui.section("Explicabilidade global")
        ui.step(f"Calculando importancia de features ({xai.method_name()})...")
        importance, importance_method = xai.global_importance(
            trained_object, X[idx_train], feature_names, top_k=top_k,
            max_samples=200 if X.shape[1] > 120 else 300,
        )
        if importance:
            # a leitura usa o valor TIPICO (mediana) da feature entre as fraudes do
            # treino - assim a tabela global explica o padrao, nao uma linha sorteada
            fraud_rows = idx_train[y_train == 1]
            reference_idx = int(fraud_rows[0]) if len(fraud_rows) else int(idx_train[0])
            medians = raw.iloc[fraud_rows if len(fraud_rows) else idx_train].median(numeric_only=True)
            body = []
            for position, (name, value) in enumerate(importance, start=1):
                typical = float(medians.get(name, 0.0))
                reading = module.humanize(name, typical, value, frame.iloc[reference_idx])
                body.append([position, name, f"{value:.5f}",
                             reading if len(reading) <= 72 else reading[:69] + "..."])
            ui.table(
                ["#", "feature", "influencia", "leitura (valor tipico nas fraudes)"],
                body,
                title=f"Top {len(importance)} features - metodo: {importance_method}",
            )
            path = xai.plot_global_importance(
                importance, report_dir(module.key) / f"{kind}_importancia_global.png",
                f"FraudShield BR - {module.title} ({importance_method})",
            )
            if path:
                ui.info(f"grafico de importancia: {Path(path).name}")

    # ------------------------------------------------------------------ saida
    bundle["metrics"] = metrics
    bundle["smote"] = smote_info
    bundle["train_seconds"] = train_seconds
    bundle["model_label"] = model_label
    bundle["importance"] = importance

    reports = evaluate.save_report(
        metrics, report_dir(module.key), module.key, model_label, importance,
        extra={"smote": smote_info, "treino_segundos": round(train_seconds, 3),
               "graficos": plots, "fonte_dados": bundle["source"]},
    )

    if save:
        destination = model_path(module.key, kind)
        if kind == "bert":
            trained_object.save(destination)
            joblib.dump(
                {k: v for k, v in bundle.items() if k not in ("model", "background")},
                Path(destination) / "bundle.joblib",
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, destination, compress=3)
        ui.ok(f"modelo salvo em {destination}")
    ui.info(f"relatorio: {Path(reports['markdown']).relative_to(Path.cwd()) if Path(reports['markdown']).is_relative_to(Path.cwd()) else reports['markdown']}")

    return {
        "bundle": bundle,
        "metrics": metrics,
        "scores": scores,
        "y_test": y_test,
        "idx_test": idx_test,
        "idx_train": idx_train,
        "X": X,
        "raw": raw,
        "frame": frame,
        "model": trained_object,
        "model_label": model_label,
        "importance": importance,
        "plots": plots,
        "reports": reports,
        "train_seconds": train_seconds,
    }
