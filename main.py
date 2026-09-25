#!/usr/bin/env python3
"""FraudShield BR - entrypoint.

    python main.py --module transaction --demo
    python main.py --module email --predict data/samples/exemplo_phishing.eml
    python main.py --module social --train --model bert
    python main.py --module transaction --evaluate

Uso exclusivamente defensivo: times de seguranca e antifraude.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
