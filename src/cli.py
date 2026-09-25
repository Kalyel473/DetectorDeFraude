"""Interface de linha de comando do FraudShield BR.

    python main.py --module transaction --demo
    python main.py --module email --predict data/samples/exemplo_phishing.eml
    python main.py --module social --train --model bert
    python main.py --module transaction --evaluate
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import __version__, ui
from .config import SEED, TEST_SIZE, TOP_FACTORS, ensure_dirs, model_path, report_dir
from .models.baseline_sklearn import MODEL_SHORT
from .modules import MODULES, get_module, module_table
from .pipeline import evaluate as ev
from .pipeline import predict as pr
from .pipeline import train as tr

AVISO_LEGAL = (
    "Ferramenta de uso exclusivamente DEFENSIVO (times de seguranca/antifraude). "
    "Datasets de exemplo sao sinteticos ou publicos e anonimizados. "
    "Em producao com dados reais: LGPD (Lei 13.709/2018) e Lei 12.737/2012."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraudshield",
        description="FraudShield BR - deteccao de fraude com Machine Learning (uso defensivo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exemplos:\n"
            "  python main.py --module transaction --demo\n"
            "  python main.py --module email --demo\n"
            "  python main.py --module social --demo\n"
            "  python main.py --module email --predict data/samples/exemplo_phishing.eml\n"
            "  python main.py --module transaction --train --model xgb --data data/raw/creditcard.csv\n"
            "  python main.py --module transaction --evaluate\n"
            "  python main.py --module email --compare\n"
            "  python main.py --module social --train --model bert --bert-epochs 2\n"
        ),
    )
    parser.add_argument("-m", "--module", help=f"modulo: {', '.join(MODULES)}")

    actions = parser.add_argument_group("acoes")
    actions.add_argument("--demo", action="store_true",
                         help="pipeline completo com a amostra do repositorio (sem setup)")
    actions.add_argument("--train", action="store_true", help="treina e salva o modelo")
    actions.add_argument("--evaluate", action="store_true",
                         help="avalia o modelo salvo e gera relatorio de metricas")
    actions.add_argument("--predict", metavar="ARQUIVO",
                         help="classifica um arquivo (.eml, .csv, .json)")
    actions.add_argument("--compare", nargs="?", const="rf,gb,logreg", metavar="MODELOS",
                         help="treina varios modelos e imprime a tabela comparativa")
    actions.add_argument("--list-modules", action="store_true", help="lista os modulos disponiveis")
    actions.add_argument("--version", action="version", version=f"FraudShield BR {__version__}")

    data = parser.add_argument_group("dados e modelo")
    data.add_argument("--data", metavar="ARQUIVO", help="dataset de treino (CSV/JSON/Parquet)")
    data.add_argument("--model", default=None,
                      help="rf (padrao), gb, xgb, logreg ou bert")
    data.add_argument("--no-smote", action="store_true", help="desliga o SMOTE (para comparacao)")
    data.add_argument("--test-size", type=float, default=TEST_SIZE, help=f"padrao {TEST_SIZE}")
    data.add_argument("--seed", type=int, default=SEED, help=f"semente aleatoria (padrao {SEED})")
    data.add_argument("--whois", action="store_true",
                      help="modulo email: consulta WHOIS real da idade do dominio (rede)")

    out = parser.add_argument_group("saida")
    out.add_argument("--factors", type=int, default=TOP_FACTORS,
                     help=f"fatores por predicao (padrao {TOP_FACTORS})")
    out.add_argument("--top-k", type=int, default=15, help="features na importancia global")
    out.add_argument("--limit", type=int, default=5,
                     help="quantos itens explicar (0 = todos)")
    out.add_argument("--threshold", type=float, default=None, help="limiar de decisao (padrao 0.5)")
    out.add_argument("--only-flagged", action="store_true", help="mostra apenas itens acima do limiar")
    out.add_argument("--json", metavar="ARQUIVO", help="salva o resultado da predicao em JSON")
    out.add_argument("--no-plots", action="store_true", help="nao gera PNGs")
    out.add_argument("--no-shap", action="store_true", help="pula a importancia global (mais rapido)")
    out.add_argument("--no-banner", action="store_true", help="saida sem banner")

    bert = parser.add_argument_group("camada BERT (opcional)")
    bert.add_argument("--bert-epochs", type=int, default=2)
    bert.add_argument("--bert-batch", type=int, default=16)
    bert.add_argument("--bert-model", default=None,
                      help="padrao: neuralmind/bert-base-portuguese-cased (BERTimbau)")
    return parser


def list_modules() -> None:
    ui.section("Modulos disponiveis")
    ui.table(["--module", "dominio", "features principais"], module_table())
    ui.raw()
    ui.info("cada modulo aceita --demo, --train, --evaluate, --predict e --compare")


def _make_module(args):
    kwargs = {}
    if args.module and args.module.lower().startswith("email"):
        kwargs["use_whois"] = args.whois
    return get_module(args.module, **kwargs)


def _explain_top(module, bundle, frame, positions, args, title: str) -> None:
    """Explica as N linhas de maior score (usado no fim do --demo)."""
    if not positions:
        ui.info("nenhum item classificado como fraude no conjunto de teste.")
        return
    ui.section(title)
    subset = frame.iloc[positions].reset_index(drop=True)
    results = pr.run_predict(
        module, bundle, subset,
        top_k=args.factors,
        limit=None,
        threshold=args.threshold,
        json_out=args.json,
    )
    pr.summarize(results, module)


def cmd_demo(module, args) -> int:
    ui.info("modo --demo: amostra sintetica do repositorio, sem download e sem GPU")
    result = tr.run_train(
        module,
        data_path=args.data,
        demo=True,
        model_kind=args.model,
        use_smote=not args.no_smote,
        test_size=args.test_size,
        seed=args.seed,
        save=True,
        make_plots=not args.no_plots,
        explain=not args.no_shap,
        top_k=args.top_k,
        bert_epochs=args.bert_epochs,
        bert_batch=args.bert_batch,
        bert_model=args.bert_model,
    )
    bundle = result["bundle"]
    scores = result["scores"]
    idx_test = result["idx_test"]
    limit = args.limit if args.limit and args.limit > 0 else 5
    order = np.argsort(scores)[::-1][:limit]
    positions = [int(idx_test[i]) for i in order]
    _explain_top(module, bundle, result["frame"], positions, args,
                 "Predicao explicada - itens de maior score no conjunto de teste")
    ui.raw()
    ui.info("proximos passos: --train com seu dataset (--data), --evaluate para o relatorio, "
            "--predict para classificar um arquivo")
    return 0


def cmd_train(module, args) -> int:
    tr.run_train(
        module,
        data_path=args.data,
        demo=args.demo,
        model_kind=args.model,
        use_smote=not args.no_smote,
        test_size=args.test_size,
        seed=args.seed,
        save=True,
        make_plots=not args.no_plots,
        explain=not args.no_shap,
        top_k=args.top_k,
        bert_epochs=args.bert_epochs,
        bert_batch=args.bert_batch,
        bert_model=args.bert_model,
    )
    return 0


def cmd_evaluate(module, args) -> int:
    kind = (args.model or module.default_model).lower()
    try:
        bundle = pr.load_bundle(module.key, kind)
    except pr.ModelNotTrained as exc:
        ui.warn(str(exc))
        ui.step("treinando agora para poder avaliar...")
        return cmd_train(module, args)

    ui.section(f"Avaliacao do modelo salvo - {bundle.get('model_label', kind)}")
    ui.kv([
        ("modulo", module.key),
        ("modelo", bundle.get("model_label", kind)),
        ("treinado em", bundle.get("created_at", "?")),
        ("fonte dos dados", bundle.get("source", "?")),
        ("SMOTE", bundle.get("smote", {}).get("backend") or bundle.get("smote", {}).get("reason", "-")),
    ])

    frame = module.load_data(args.data, demo=args.demo)
    scores, _, _ = pr.score_frame(module, bundle, frame)
    label_col = bundle.get("artifacts", {}).get("label_col") or "label"
    if label_col not in frame.columns:
        ui.err(f"o dataset avaliado nao tem a coluna de rotulo '{label_col}'.")
        return 2
    y = frame[label_col].astype(int).to_numpy()

    # reproduz o mesmo split do treino para avaliar apenas dados nao vistos
    if bundle.get("n_train", 0) + bundle.get("n_test", 0) == len(y):
        idx_train, idx_test = tr._split(len(y), y, args.test_size, args.seed)
        ui.info(f"avaliando no mesmo conjunto de teste do treino ({len(idx_test)} registros, seed {args.seed})")
    else:
        idx_test = np.arange(len(y))
        ui.warn("tamanho do dataset difere do treino: avaliando a base inteira")

    metrics = ev.compute_metrics(y[idx_test], scores[idx_test],
                                threshold=args.threshold or bundle.get("threshold", 0.5))
    ev.print_metrics(metrics, "Metricas recalculadas")
    plots = {} if args.no_plots else ev.plot_curves(
        y[idx_test], scores[idx_test], report_dir(module.key), prefix=f"{kind}_eval_",
    )
    reports = ev.save_report(metrics, report_dir(module.key), module.key,
                             bundle.get("model_label", kind), bundle.get("importance"),
                             extra={"graficos": plots, "modo": "evaluate"})
    ui.info(f"relatorio: {reports['markdown']}")
    if bundle.get("metrics"):
        old = bundle["metrics"]
        ev.comparison_table([
            {"modelo": "salvas no treino", "precision": old["precision"], "recall": old["recall"],
             "f1": old["f1"], "pr_auc": old.get("pr_auc"), "roc_auc": old.get("roc_auc"),
             "treino_s": bundle.get("train_seconds", 0), "custo": "-"},
            {"modelo": "recalculadas agora", "precision": metrics["precision"],
             "recall": metrics["recall"], "f1": metrics["f1"], "pr_auc": metrics.get("pr_auc"),
             "roc_auc": metrics.get("roc_auc"), "treino_s": 0, "custo": "-"},
        ], title="Conferencia de reprodutibilidade")
    return 0


def cmd_predict(module, args) -> int:
    kind = (args.model or module.default_model).lower()
    try:
        bundle = pr.load_bundle(module.key, kind)
    except pr.ModelNotTrained as exc:
        ui.err(str(exc))
        ui.info("dica: rode primeiro `python main.py --module "
                f"{module.key} --demo` para treinar com a amostra inclusa")
        return 2

    ui.section(f"Analisando {args.predict}")
    frame = module.parse_input(args.predict)
    ui.ok(f"{len(frame)} item(ns) para classificar | modelo: {bundle.get('model_label', kind)}")
    for note in module.input_notes(frame):
        ui.warn(note)
    results = pr.run_predict(
        module, bundle, frame,
        top_k=args.factors,
        limit=None if args.limit == 0 else args.limit,
        only_flagged=args.only_flagged,
        threshold=args.threshold,
        json_out=args.json,
    )
    pr.summarize(results, module)
    return 0


def cmd_compare(module, args) -> int:
    kinds = [k.strip() for k in args.compare.split(",") if k.strip()]
    ui.info(f"treinando {len(kinds)} modelos para comparacao: {', '.join(kinds)}")
    rows = []
    for kind in kinds:
        started = time.perf_counter()
        try:
            result = tr.run_train(
                module,
                data_path=args.data,
                demo=args.demo or not args.data,
                model_kind=kind,
                use_smote=not args.no_smote,
                test_size=args.test_size,
                seed=args.seed,
                save=True,
                make_plots=not args.no_plots,
                explain=False,
                bert_epochs=args.bert_epochs,
                bert_batch=args.bert_batch,
                bert_model=args.bert_model,
            )
        except SystemExit as exc:
            ui.warn(f"{kind}: {exc}")
            continue
        metrics = result["metrics"]
        real_kind = result["bundle"].get("model_kind", kind)
        label = MODEL_SHORT.get(real_kind, real_kind)
        if real_kind != kind:
            label += f" (fallback de {kind})"
        rows.append({
            "modelo": label,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "pr_auc": metrics.get("pr_auc"),
            "roc_auc": metrics.get("roc_auc"),
            "treino_s": time.perf_counter() - started,
            "custo": "GPU recomendada" if kind == "bert" else "CPU",
        })
    if rows:
        ev.comparison_table(rows, title=f"Baseline sklearn x BERT - modulo {module.key}")
        ui.info("trade-off: o baseline treina em segundos na CPU; o BERT exige GPU e "
                "minutos de fine-tuning para ganhar alguns pontos de recall em texto.")
    return 0 if rows else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_dirs()

    if not args.no_banner:
        ui.banner(__version__)
    ui.raw(f"[dim]{AVISO_LEGAL}[/dim]")
    ui.raw()

    if args.list_modules or not args.module:
        list_modules()
        if not args.module:
            ui.raw()
            ui.warn("informe um modulo: --module transaction|email|social  (veja --help)")
            return 0 if args.list_modules else 1
        return 0

    try:
        module = _make_module(args)
    except KeyError as exc:
        ui.err(str(exc))
        return 2

    try:
        if args.compare:
            return cmd_compare(module, args)
        if args.predict:
            return cmd_predict(module, args)
        if args.evaluate:
            return cmd_evaluate(module, args)
        if args.train:
            return cmd_train(module, args)
        if args.demo:
            return cmd_demo(module, args)
        ui.warn("nenhuma acao informada. Use --demo, --train, --evaluate, --predict ou --compare.")
        ui.kv([
            ("modulo", f"{module.key} - {module.title}"),
            ("modelo padrao", module.default_model),
            ("amostra do modo demo", f"data/samples/{module.sample_file}"),
            ("modelo salvo", str(model_path(module.key, module.default_model))),
            ("dataset real", module.dataset_hint),
        ])
        return 1
    except FileNotFoundError as exc:
        ui.err(str(exc))
        return 2
    except KeyboardInterrupt:
        ui.warn("interrompido pelo usuario")
        return 130


if __name__ == "__main__":
    sys.exit(main())
