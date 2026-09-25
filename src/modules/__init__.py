"""Registro dos modulos de deteccao do FraudShield BR."""
from __future__ import annotations

from .base import FraudModule
from .email_phishing import EmailPhishingModule
from .social_fraud import SocialFraudModule
from .transaction_fraud import TransactionFraudModule

MODULES = {
    "transaction": TransactionFraudModule,
    "email": EmailPhishingModule,
    "social": SocialFraudModule,
}

ALIASES = {
    "transacao": "transaction",
    "transacoes": "transaction",
    "tx": "transaction",
    "card": "transaction",
    "phishing": "email",
    "mail": "email",
    "e-mail": "email",
    "twitter": "social",
    "redes": "social",
    "tweets": "social",
}


def get_module(name: str, **kwargs) -> FraudModule:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in MODULES:
        raise KeyError(
            f"Modulo '{name}' nao existe. Disponiveis: {', '.join(MODULES)}."
        )
    cls = MODULES[key]
    try:
        return cls(**kwargs)
    except TypeError:
        return cls()


def module_table() -> list[tuple[str, str, str]]:
    return [(key, cls.title, cls.description) for key, cls in MODULES.items()]


__all__ = ["MODULES", "ALIASES", "get_module", "module_table", "FraudModule"]
