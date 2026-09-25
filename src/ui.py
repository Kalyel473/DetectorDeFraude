"""Camada de apresentacao no terminal (estilo 'Terminal Verde').

Usa `rich` quando disponivel e degrada para print() puro caso contrario,
para que a ferramenta nunca quebre por causa de dependencia de UI.
"""
from __future__ import annotations

import sys
from typing import Iterable, Sequence

# --- compatibilidade de encoding no Windows (cp1252 nao tem setas/caixas) ---
try:  # pragma: no cover - depende do console
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _supports(char: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        char.encode(enc)
        return True
    except Exception:
        return False


ARROW = "\u2192" if _supports("\u2192") else "->"
BULLET = "\u2022" if _supports("\u2022") else "*"

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme

    _THEME = Theme(
        {
            "ok": "bold green",
            "info": "green",
            "dim": "grey62",
            "warn": "bold yellow",
            "err": "bold red",
            "high": "bold red",
            "mid": "bold yellow",
            "low": "bold cyan",
            "clean": "bold green",
            "head": "bold bright_green",
        }
    )
    # saida redirecionada (pipe/arquivo) ganha largura fixa para a tabela nao
    # ser comprimida a ponto de truncar os nomes das metricas
    _tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    console = Console(theme=_THEME, highlight=False, width=None if _tty else 118)
    RICH = True
except Exception:  # pragma: no cover
    console = None
    RICH = False


BANNER = r"""
  ______                 _ ____  _     _      _     _
 |  ____|               | / ___|| |__ (_) ___| | __| |
 | |__ _ __ __ _ _   _ _| \___ \| '_ \| |/ _ \ |/ _` |
 |  __| '__/ _` | | | / _` |__) | | | | |  __/ | (_| |
 | |  | | | (_| | |_| \__,_|___/|_| |_|_|\___|_|\__,_|
 |_|  |_|  \__,_|\__,_|  F R A U D S H I E L D   B R
"""

STYLE_BY_RISK = {
    "ALTO RISCO": "high",
    "RISCO MEDIO": "mid",
    "RISCO BAIXO": "low",
    "SEM INDICIO": "clean",
}


def _plain(text: str) -> str:
    """Remove marcacao rich para o modo degradado."""
    out, depth = [], 0
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
        elif not depth:
            out.append(ch)
    return "".join(out)


def raw(text: str = "", style: str | None = None) -> None:
    if RICH:
        console.print(text, style=style)
    else:
        print(_plain(text))


def banner(version: str, subtitle: str = "Detector de Fraude com IA - uso defensivo") -> None:
    if RICH:
        console.print(BANNER, style="bold green")
        console.print(f"  FraudShield BR v{version} {BULLET} {subtitle}", style="dim")
        console.print()
    else:
        print(BANNER)
        print(f"  FraudShield BR v{version} - {subtitle}\n")


def section(title: str) -> None:
    if RICH:
        console.rule(f"[head]{title}[/head]", style="green")
    else:
        print(f"\n=== {title} ===")


def step(text: str) -> None:
    raw(f"[info][FRAUDSHIELD BR][/info] {text}")


def ok(text: str) -> None:
    raw(f"[ok][ + ][/ok] {text}")


def info(text: str) -> None:
    raw(f"[info][ i ][/info] {text}")


def warn(text: str) -> None:
    raw(f"[warn][ ! ][/warn] {text}")


def err(text: str) -> None:
    raw(f"[err][ x ][/err] {text}")


def arrow(text: str, style: str = "info") -> None:
    raw(f"[{style}]{ARROW}[/{style}] {text}")


def bullets(items: Iterable[str], indent: str = "  ") -> None:
    for item in items:
        raw(f"{indent}{BULLET} {item}")


def kv(pairs: Sequence[tuple], title: str | None = None) -> None:
    if RICH:
        tbl = Table(show_header=False, box=None, pad_edge=False, title=title)
        tbl.add_column(style="dim")
        tbl.add_column(style="bold")
        for key, value in pairs:
            tbl.add_row(str(key), str(value))
        console.print(tbl)
    else:
        if title:
            print(title)
        for key, value in pairs:
            print(f"  {key}: {value}")


def table(headers: Sequence[str], rows: Sequence[Sequence], title: str | None = None) -> None:
    if RICH:
        tbl = Table(title=title, header_style="head", border_style="green")
        for head in headers:
            tbl.add_column(str(head))
        for row in rows:
            tbl.add_row(*[str(cell) for cell in row])
        console.print(tbl)
    else:
        if title:
            print(f"\n{title}")
        print(" | ".join(str(h) for h in headers))
        for row in rows:
            print(" | ".join(str(c) for c in row))


def confusion(matrix, labels: Sequence[str] = ("LEGITIMO", "FRAUDE")) -> None:
    """Matriz de confusao 2x2 formatada para leitura rapida na demo."""
    (tn, fp), (fn, tp) = matrix
    rows = [
        [f"real {labels[0]}", f"{tn}  (TN)", f"{fp}  (FP - falso alarme)"],
        [f"real {labels[1]}", f"{fn}  (FN - fraude perdida)", f"{tp}  (TP)"],
    ]
    table(["", f"previsto {labels[0]}", f"previsto {labels[1]}"], rows, title="Matriz de confusao")


def score_panel(title: str, score: float, level: str, factors: Sequence[str], recommendation: str,
                factors_title: str = "Principais fatores") -> None:
    style = STYLE_BY_RISK.get(level, "info")
    lines = [f"{ARROW} Score de fraude: {score:.2f} ({level})"]
    if factors:
        lines.append(f"{ARROW} {factors_title}:")
        lines.extend([f"    {BULLET} {f}" for f in factors])
    else:
        lines.append(f"{ARROW} Nenhum fator isolado acima do limiar de relevancia.")
    lines.append(f"{ARROW} Recomendacao: {recommendation}")
    body = "\n".join(lines)
    if RICH:
        console.print(Panel(Text(body), title=f"[{style}]{title}[/{style}]", border_style=style, expand=False))
    else:
        print(f"\n[{title}]")
        print(body)


def gauge(score: float, width: int = 30) -> str:
    filled = int(round(score * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {score:.2%}"
