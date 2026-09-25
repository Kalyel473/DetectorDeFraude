"""Predicao com explicacao: score + por que + o que fazer.

Nenhuma saida do FraudShield diz apenas "fraude". Toda decisao vem com os
fatores que a sustentam - e o mesmo padrao de transparencia usado no
ImunoShield (formula de risco visivel) e no EscalaMind (regras explicaveis).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .. import ui
from ..config import TOP_FACTORS, model_path, risk_label
from ..models import explainability as xai
from ..models.baseline_sklearn import positive_proba
from ..modules.base import FraudModule


class ModelNotTrained(RuntimeError):
    pass


def load_bundle(module_key: str, kind: str = "rf") -> dict:
    """Carrega o modelo serializado do disco (ou levanta ModelNotTrained)."""
    path = model_path(module_key, kind)
    if kind == "bert":
        meta_file = Path(path) / "bundle.joblib"
        if not meta_file.exists():
            raise ModelNotTrained(
                f"Nenhum BERT treinado em {path}. Rode: "
                f"python main.py --module {module_key} --train --model bert"
            )
        from ..models.bert_classifier import BertTextClassifier

        bundle = joblib.load(meta_file)
        bundle["bert"] = BertTextClassifier.load(path)
        return bundle
    if not path.exists():
        raise ModelNotTrained(
            f"Modelo nao encontrado em {path}. Rode primeiro: "
            f"python main.py --module {module_key} --train --model {kind}"
        )
    return joblib.load(path)


def score_frame(module: FraudModule, bundle: dict, frame: pd.DataFrame):
    """Retorna (scores, X, raw_frame). Usa os artefatos salvos no treino."""
    frame = frame.reset_index(drop=True)
    if bundle.get("model_kind") == "bert":
        texts = module.texts(frame)
        scores = bundle["bert"].predict_proba(texts)[:, 1]
        return np.asarray(scores, dtype=float), None, None
    X, _, _, _, raw = module.build_features(frame, bundle["artifacts"])
    scores = positive_proba(bundle["model"], X)
    return np.asarray(scores, dtype=float), X, raw


def explain_rows(module: FraudModule, bundle: dict, frame: pd.DataFrame, X, raw,
                 positions: list[int], top_k: int = TOP_FACTORS) -> dict:
    """Fatores por linha, ja traduzidos para portugues."""
    explanations: dict[int, tuple[list[str], str]] = {}
    feature_names = bundle["feature_names"]
    background = bundle.get("background")

    if bundle.get("model_kind") == "bert":
        from ..models.bert_classifier import explain_by_occlusion

        texts = module.texts(frame)
        for position in positions:
            words = explain_by_occlusion(bundle["bert"], texts[position], top_k=top_k)
            factors = [f'termo "{word}" puxa a classificacao para fraude (+{delta:.3f})'
                       for word, delta in words]
            explanations[position] = (factors, "oclusao de palavras (BERT)")
        return explanations

    for position in positions:
        contributions, method = xai.instance_contributions(
            bundle["model"], X[position], feature_names, background=background, top_k=top_k
        )
        row = frame.iloc[position]
        factors = []
        for name, contribution in contributions:
            value = float(raw[name].iloc[position]) if raw is not None and name in raw.columns else 0.0
            factors.append(module.humanize(name, value, contribution, row))
        explanations[position] = (factors, method)
    return explanations


def run_predict(
    module: FraudModule,
    bundle: dict,
    frame: pd.DataFrame,
    top_k: int = TOP_FACTORS,
    limit: int | None = None,
    only_flagged: bool = False,
    threshold: float | None = None,
    json_out: str | None = None,
    quiet: bool = False,
) -> list[dict]:
    frame = frame.reset_index(drop=True)
    threshold = bundle.get("threshold", 0.5) if threshold is None else threshold
    scores, X, raw = score_frame(module, bundle, frame)

    order = np.argsort(scores)[::-1]
    if only_flagged:
        order = [i for i in order if scores[i] >= threshold]
    if limit:
        order = list(order)[:limit]
    positions = [int(i) for i in order]

    explanations = explain_rows(module, bundle, frame, X, raw, positions, top_k=top_k)

    label_col = bundle.get("artifacts", {}).get("label_col")
    results = []
    for position in positions:
        row = frame.iloc[position]
        score = float(scores[position])
        level = risk_label(score)
        factors, method = explanations.get(position, ([], "-"))
        evidence = module.extra_evidence(row) if hasattr(module, "extra_evidence") else []
        item = {
            "posicao": int(position),
            "identificacao": module.describe(position, row),
            "score": round(score, 4),
            "nivel": level,
            "decisao": module.positive_name if score >= threshold else module.negative_name,
            "fatores": factors,
            "evidencias": evidence,
            "metodo_explicacao": method,
            "recomendacao": module.recommend(score),
        }
        if label_col and label_col in frame.columns:
            item["rotulo_real"] = int(row[label_col])
        results.append(item)

        if not quiet:
            flagged = score >= threshold
            ui.score_panel(
                f"{item['identificacao']} {ui.ARROW} {item['decisao']}",
                score, level, factors + evidence, item["recomendacao"],
                factors_title="Principais fatores" if flagged
                else "Sinais observados (nenhum decisivo)",
            )

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ui.info(f"resultado em JSON: {json_out}")
    return results


def summarize(results: list[dict], module: FraudModule) -> None:
    if not results:
        ui.info("nenhum item acima do limiar.")
        return
    flagged = [r for r in results if r["decisao"] == module.positive_name]
    ui.section("Resumo da analise")
    ui.kv([
        ("itens analisados", len(results)),
        (f"classificados como {module.positive_name}", len(flagged)),
        ("score maximo", f"{max(r['score'] for r in results):.4f}"),
        ("explicacao via", results[0]["metodo_explicacao"]),
    ])
    if any("rotulo_real" in r for r in results):
        hits = sum(1 for r in results if r.get("rotulo_real") == 1 and r["decisao"] == module.positive_name)
        total_real = sum(1 for r in results if r.get("rotulo_real") == 1)
        ui.info(f"conferencia com o rotulo real da amostra: {hits}/{total_real} fraudes confirmadas")
