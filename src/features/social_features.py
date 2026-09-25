
from __future__ import annotations

import hashlib
import re
import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .text_features import (
    PT_STOPWORDS,
    SHORTENERS,
    SUSPICIOUS_TLDS,
    TEXT_PREFIX,
    extract_urls,
    host_of,
    normalize_text,
    registrable,
)

SCAM_TRIGGERS = (
    "renda extra", "ganhe dinheiro", "clique no link", "link na bio", "me chama no direct",
    "chama no whats", "investimento", "lucro garantido", "dobre seu", "sinal gratis",
    "bonus", "cupom", "sorteio", "premio", "vagas limitadas", "ultimas vagas",
    "trabalhe de casa", "aposta", "bet", "cripto", "bitcoin", "airdrop", "seguidores gratis",
    "follow back", "siga de volta", "dm me", "free giveaway", "double your",
)

EMOJI_RE = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "]"
)

BURST_EPS_SECONDS = 6 * 3600.0     # contas criadas na mesma janela de 6h
BURST_MIN_SAMPLES = 3

NUMERIC_FEATURES = [
    "account_age_days", "followers", "following", "statuses_count",
    "log_followers", "log_following", "followers_following_ratio",
    "posts_per_day", "followers_per_day", "default_profile_image", "verified",
    "bio_empty", "bio_len", "bio_has_url", "username_digit_ratio", "username_len",
    "name_has_digits", "text_len", "caps_ratio", "exclamations", "emoji_count",
    "hashtags", "mentions", "n_links", "has_shortener", "has_suspicious_tld",
    "scam_triggers", "duplicate_text_count", "creation_burst_size",
]

FOLLOWER_CANDIDATES = ("followers", "followers_count", "seguidores")
FOLLOWING_CANDIDATES = ("following", "friends_count", "following_count", "seguindo")
STATUS_CANDIDATES = ("statuses_count", "posts", "tweets", "n_posts")
LABEL_CANDIDATES = ("label", "class", "is_fake", "bot", "fraude")


def find_label_col(df: pd.DataFrame) -> str | None:
    for name in LABEL_CANDIDATES:
        if name in df.columns:
            return name
    return None


def _num(row, candidates, default: float = 0.0) -> float:
    for name in candidates:
        if name in row.index:
            try:
                value = float(row[name])
                if not np.isnan(value):
                    return value
            except (TypeError, ValueError):
                continue
    return default


def _text_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", text.lower())).strip()
    norm = re.sub(r"[^a-z0-9 ]", "", norm)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest() if norm else ""


def created_epoch(df: pd.DataFrame) -> np.ndarray:
    """Timestamp de criacao da conta em segundos (0 quando ausente)."""
    for col in ("created_at", "account_created_at", "criado_em"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            epoch = parsed.astype("int64").to_numpy(dtype=float) / 1e9
            return np.where(np.isnan(epoch) | (epoch < 0), 0.0, epoch)
    return np.zeros(len(df))


def creation_bursts(epochs: np.ndarray) -> np.ndarray:
    """Tamanho do cluster temporal de criacao de cada conta (1 = isolada)."""
    sizes = np.ones(len(epochs))
    valid = epochs > 0
    if valid.sum() < BURST_MIN_SAMPLES:
        return sizes
    labels = DBSCAN(eps=BURST_EPS_SECONDS, min_samples=BURST_MIN_SAMPLES).fit_predict(
        epochs[valid].reshape(-1, 1)
    )
    counts = {lab: int((labels == lab).sum()) for lab in set(labels) if lab != -1}
    sizes[valid] = [counts.get(lab, 1) for lab in labels]
    return sizes


def row_features(row) -> dict:
    """Sinais de uma conta/post isolada (sem contexto de lote)."""
    feats: dict[str, float] = {}
    age = _num(row, ("account_age_days", "idade_conta_dias"), default=-1.0)
    followers = _num(row, FOLLOWER_CANDIDATES)
    following = _num(row, FOLLOWING_CANDIDATES)
    statuses = _num(row, STATUS_CANDIDATES)

    feats["account_age_days"] = age
    feats["followers"] = followers
    feats["following"] = following
    feats["statuses_count"] = statuses
    feats["log_followers"] = float(np.log1p(max(followers, 0)))
    feats["log_following"] = float(np.log1p(max(following, 0)))
    feats["followers_following_ratio"] = float(followers / max(following, 1.0))
    safe_age = max(age, 1.0) if age > 0 else 1.0
    feats["posts_per_day"] = float(statuses / safe_age)
    feats["followers_per_day"] = float(followers / safe_age)
    feats["default_profile_image"] = float(bool(_num(row, ("default_profile_image", "foto_padrao"))))
    feats["verified"] = float(bool(_num(row, ("verified", "verificado"))))

    bio = normalize_text(row.get("bio") or row.get("description"))
    feats["bio_empty"] = float(len(bio.strip()) == 0)
    feats["bio_len"] = float(len(bio))
    feats["bio_has_url"] = float(bool(extract_urls(bio)))

    username = normalize_text(row.get("username") or row.get("screen_name") or row.get("user"))
    digits = sum(c.isdigit() for c in username)
    feats["username_digit_ratio"] = float(digits / max(len(username), 1))
    feats["username_len"] = float(len(username))
    name = normalize_text(row.get("name") or row.get("display_name"))
    feats["name_has_digits"] = float(any(c.isdigit() for c in name))

    text = normalize_text(row.get("post_text") or row.get("text") or row.get("tweet"))
    letters = [c for c in text if c.isalpha()]
    feats["text_len"] = float(len(text))
    feats["caps_ratio"] = float(sum(c.isupper() for c in letters) / max(len(letters), 1))
    feats["exclamations"] = float(text.count("!"))
    feats["emoji_count"] = float(len(EMOJI_RE.findall(text)))
    feats["hashtags"] = float(text.count("#"))
    feats["mentions"] = float(text.count("@"))

    urls = extract_urls(text) + extract_urls(bio)
    hosts = [host_of(u) for u in urls]
    feats["n_links"] = float(len(urls))
    feats["has_shortener"] = float(any(registrable(h) in SHORTENERS for h in hosts))
    feats["has_suspicious_tld"] = float(
        any(h.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS for h in hosts if "." in h)
    )
    low = text.lower()
    feats["scam_triggers"] = float(sum(low.count(t) for t in SCAM_TRIGGERS))
    return feats


class SocialFeatureSpace:
    """TF-IDF do post + sinais comportamentais + sinais de rede (lote/treino)."""

    def __init__(self, max_features: int = 250):
        self.max_features = max_features
        self.vectorizer: TfidfVectorizer | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        # memoria do treino, usada quando a predicao recebe uma conta isolada
        self.dup_map: dict[str, int] = {}
        self.burst_map: dict[str, int] = {}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _texts(df: pd.DataFrame) -> list[str]:
        return [
            normalize_text(row.get("post_text") or row.get("text") or row.get("tweet"))
            + " "
            + normalize_text(row.get("bio") or row.get("description"))
            for _, row in df.iterrows()
        ]

    def _numeric_frame(self, df: pd.DataFrame, learn: bool) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        rows = [row_features(row) for _, row in df.iterrows()]
        frame = pd.DataFrame(rows).reset_index(drop=True)
        epochs = created_epoch(df)

        # --- idade da conta derivada de created_at quando nao vem pronta ---
        missing = (frame["account_age_days"] <= 0).to_numpy()
        if missing.any() and (epochs > 0).any():
            reference = max(time.time(), float(epochs.max()))
            ages = np.maximum((reference - epochs) / 86400.0, 0.0)
            fixed = np.where(missing & (epochs > 0), ages, frame["account_age_days"].to_numpy())
            frame["account_age_days"] = fixed
            safe_age = np.maximum(frame["account_age_days"].to_numpy(), 1.0)
            frame["posts_per_day"] = frame["statuses_count"].to_numpy() / safe_age
            frame["followers_per_day"] = frame["followers"].to_numpy() / safe_age

        # --- texto repetido entre contas (spam coordenado) ---
        hashes = [_text_hash(t) for t in self._texts(df)]
        batch_counts: dict[str, int] = {}
        for h in hashes:
            if h:
                batch_counts[h] = batch_counts.get(h, 0) + 1
        if learn:
            for h, c in batch_counts.items():
                self.dup_map[h] = max(self.dup_map.get(h, 0), c)
        frame["duplicate_text_count"] = [
            float(max(batch_counts.get(h, 1), self.dup_map.get(h, 1))) if h else 1.0
            for h in hashes
        ]

        # --- rajada de criacao de contas (fazenda de bots) ---
        burst = creation_bursts(epochs)
        days = [
            pd.Timestamp(e, unit="s", tz="UTC").strftime("%Y-%m-%d") if e > 0 else ""
            for e in epochs
        ]
        if learn:
            day_counts: dict[str, int] = {}
            for day in days:
                if day:
                    day_counts[day] = day_counts.get(day, 0) + 1
            for day, count in day_counts.items():
                self.burst_map[day] = max(self.burst_map.get(day, 0), count)
        frame["creation_burst_size"] = [
            float(max(b, self.burst_map.get(day, 1))) for b, day in zip(burst, days)
        ]
        return frame.reindex(columns=NUMERIC_FEATURES).astype(float).fillna(0.0)

    # ------------------------------------------------------------------- api
    def fit(self, df: pd.DataFrame) -> "SocialFeatureSpace":
        self.fit_transform(df)
        return self

    def fit_transform(self, df: pd.DataFrame):
        """Ajusta e ja devolve a matriz - evita extrair as features duas vezes."""
        texts = self._texts(df)
        min_df = 2 if len(texts) > 40 else 1
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=min_df,
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words=PT_STOPWORDS,
        ).fit(texts)
        numeric = self._numeric_frame(df, learn=True)
        self.scaler = StandardScaler().fit(numeric.to_numpy(dtype=np.float32))
        self.feature_names = [TEXT_PREFIX + t for t in self.vectorizer.get_feature_names_out()]
        self.feature_names = self.feature_names + list(NUMERIC_FEATURES)
        return self._assemble(texts, numeric)

    def _assemble(self, texts: list[str], numeric: pd.DataFrame):
        tfidf = self.vectorizer.transform(texts).toarray().astype(np.float32)
        scaled = self.scaler.transform(numeric.to_numpy(dtype=np.float32)).astype(np.float32)
        X = np.hstack([tfidf, scaled])
        text_names = self.feature_names[: tfidf.shape[1]]
        raw = pd.concat([pd.DataFrame(tfidf, columns=text_names), numeric], axis=1)
        return X, raw

    def transform(self, df: pd.DataFrame):
        if self.vectorizer is None or self.scaler is None:
            raise RuntimeError("SocialFeatureSpace.fit() precisa rodar antes de transform()")
        return self._assemble(self._texts(df), self._numeric_frame(df, learn=False))
