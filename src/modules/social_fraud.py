"""Modulo 3 - fraude em redes sociais (perfis falsos, bots, golpes).

Dataset de referencia: Cresci et al. (fake followers / social spambots),
referencia classica da area. Qualquer CSV com colunas de conta + texto serve:
username, name, bio, created_at, followers, following, statuses_count,
default_profile_image, verified, post_text, label (1 = perfil/post fraudulento).
"""
from __future__ import annotations

import pandas as pd

from ..config import RAW_DIR, SAMPLES_DIR
from ..features.social_features import SocialFeatureSpace, find_label_col
from ..features.text_features import TEXT_PREFIX, normalize_text
from .base import FraudModule, read_table, text_term_reason

SAMPLE_NAME = "social_amostra.csv"
RAW_CANDIDATES = ("social.csv", "twitter_accounts.csv", "cresci.csv")


class SocialFraudModule(FraudModule):
    key = "social"
    title = "Fraude em redes sociais (perfis e posts)"
    description = "Sinais comportamentais da conta + texto do post + rajada de criacao (rede)"
    sample_file = SAMPLE_NAME
    default_model = "rf"
    positive_name = "FRAUDULENTO"
    negative_name = "AUTENTICO"
    dataset_hint = (
        "Use o dataset Cresci et al. (fake followers) em data/raw/social.csv, "
        "ou --demo para a amostra do repositorio."
    )

    def __init__(self, max_text_features: int = 250):
        self.max_text_features = max_text_features

    # ------------------------------------------------------------------ dados
    def load_data(self, path: str | None = None, demo: bool = False) -> pd.DataFrame:
        if path:
            return read_table(path)
        if not demo:
            for name in RAW_CANDIDATES:
                candidate = RAW_DIR / name
                if candidate.exists():
                    return read_table(candidate)
        return read_table(SAMPLES_DIR / SAMPLE_NAME)

    def build_features(self, df: pd.DataFrame, artifacts: dict | None = None):
        df = df.reset_index(drop=True)
        fitting = artifacts is None
        artifacts = dict(artifacts or {})
        if fitting:
            space = SocialFeatureSpace(max_features=self.max_text_features)
            artifacts["space"] = space
            X, raw = space.fit_transform(df)
        else:
            space = artifacts["space"]
            X, raw = space.transform(df)
        label_col = find_label_col(df)
        y = df[label_col].astype(int).to_numpy() if label_col else None
        artifacts["feature_names"] = space.feature_names
        artifacts["label_col"] = label_col
        return X, y, space.feature_names, artifacts, raw

    def input_notes(self, df: pd.DataFrame) -> list:
        if len(df) < 20:
            return [
                "sinais de rede (rajada de criacao e texto repetido) sao de lote: com poucas "
                "contas o FraudShield recorre a memoria do treino para esses dois campos."
            ]
        return []

    def texts(self, df: pd.DataFrame) -> list[str]:
        df = df.reset_index(drop=True)
        return [
            (normalize_text(row.get("post_text") or row.get("text")) + " " +
             normalize_text(row.get("bio"))).strip()
            for _, row in df.iterrows()
        ]

    # ------------------------------------------------------------------ saida
    def describe(self, position: int, row: pd.Series) -> str:
        username = str(row.get("username", "") or row.get("screen_name", "") or "").strip()
        return f"perfil @{username}" if username else f"perfil #{position}"

    def humanize(self, feature: str, value: float, contribution: float, row: pd.Series) -> str:
        value = float(value)
        if feature.startswith(TEXT_PREFIX):
            return text_term_reason(feature, value)

        if feature == "account_age_days":
            return f"conta criada ha {value:.0f} dias"
        if feature == "followers_following_ratio":
            followers = row.get("followers", 0) if hasattr(row, "get") else 0
            following = row.get("following", 0) if hasattr(row, "get") else 0
            return (f"razao seguidores/seguindo = {value:.2f} "
                    f"({followers:.0f} seguidores para {following:.0f} seguindo)")
        if feature in ("following", "log_following"):
            return f"segue {row.get('following', value):.0f} contas"
        if feature in ("followers", "log_followers"):
            return f"tem {row.get('followers', value):.0f} seguidores"
        if feature == "posts_per_day":
            return f"{value:.1f} posts por dia (atividade automatizada acima do humano)"
        if feature == "followers_per_day":
            return f"{value:.2f} seguidores ganhos por dia"
        if feature == "statuses_count":
            return f"{value:.0f} posts no total"
        if feature == "default_profile_image":
            return "foto de perfil padrao (conta nunca personalizada)" if value else "tem foto de perfil"
        if feature == "verified":
            return "conta verificada" if value else "conta nao verificada"
        if feature == "bio_empty":
            return "biografia vazia" if value else "biografia preenchida"
        if feature == "bio_len":
            return f"biografia com {value:.0f} caracteres"
        if feature == "bio_has_url":
            return "link na biografia"
        if feature == "username_digit_ratio":
            return f"{value * 100:.0f}% do @ sao digitos (padrao de conta gerada em massa)"
        if feature == "username_len":
            return f"@ com {value:.0f} caracteres"
        if feature == "name_has_digits":
            return "nome de exibicao contem digitos"
        if feature == "duplicate_text_count":
            return f"texto identico publicado por {value:.0f} contas (spam coordenado)"
        if feature == "creation_burst_size":
            return f"{value:.0f} contas criadas na mesma janela de 6h (fazenda de bots)"
        if feature == "scam_triggers":
            return f"{value:.0f} termo(s) de golpe no post (renda extra, link na bio, aposta...)"
        if feature == "has_shortener":
            return "link encurtado no post/bio"
        if feature == "has_suspicious_tld":
            return "link com TLD de alto abuso"
        if feature == "n_links":
            return f"{value:.0f} link(s) no post"
        if feature == "caps_ratio":
            return f"{value * 100:.0f}% do post em CAIXA ALTA"
        if feature == "exclamations":
            return f"{value:.0f} pontos de exclamacao no post"
        if feature == "emoji_count":
            return f"{value:.0f} emojis no post"
        if feature == "hashtags":
            return f"{value:.0f} hashtags no post"
        if feature == "mentions":
            return f"{value:.0f} mencoes (@) no post"
        if feature == "text_len":
            return f"post com {value:.0f} caracteres"
        return f"{feature} = {value:.4g}"

    def recommend(self, score: float) -> str:
        if score >= 0.80:
            return "suspender perfil + revisao do time de integridade"
        if score >= 0.50:
            return "limitar alcance e exigir verificacao"
        if score >= 0.25:
            return "monitorar comportamento por 7 dias"
        return "nenhuma acao"
