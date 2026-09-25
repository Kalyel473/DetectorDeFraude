"""Features de texto, URL e cabecalho para deteccao de phishing em e-mail.

Tres blocos de sinal, propositalmente separados para ficarem explicaveis:
  1. TEXTO     - TF-IDF (1-2 gramas) + contagem de palavras-gatilho por categoria.
  2. URL       - IP puro, encurtador, TLD suspeito, punycode, typosquatting
                 (distancia de Levenshtein contra marcas conhecidas) e marca
                 usada em subdominio/caminho com dominio registravel de terceiro.
  3. CABECALHO - divergencia From x Reply-To, SPF/DKIM invalido, remetente
                 freemail se passando por marca, idade do dominio (WHOIS).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Vocabulario de risco
# --------------------------------------------------------------------------- #
TRIGGER_GROUPS: dict[str, tuple[str, ...]] = {
    "urgencia": (
        "urgente", "imediatamente", "ultimo aviso", "ultima chance", "agora mesmo",
        "prazo final", "expira hoje", "em 24 horas", "acao necessaria", "nao ignore",
        "urgent", "immediately", "act now", "final notice", "expires today",
    ),
    "credencial": (
        "clique aqui", "verifique agora", "confirme seus dados", "atualize seu cadastro",
        "valide sua conta", "recadastramento", "desbloqueie sua conta", "confirmar senha",
        "informe sua senha", "token", "codigo de seguranca", "click here", "verify now",
        "update your account", "confirm your password", "login para continuar",
    ),
    "ameaca": (
        "sua conta sera bloqueada", "conta suspensa", "sera cancelado", "sera excluido",
        "pendencia", "restricao", "processo judicial", "intimacao", "divida ativa",
        "protesto", "negativacao", "account suspended", "will be blocked", "legal action",
    ),
    "dinheiro": (
        "premio", "sorteio", "voce foi selecionado", "resgate seu", "credito liberado",
        "restituicao", "reembolso", "pix", "boleto", "segunda via", "fatura em atraso",
        "emprestimo aprovado", "cashback", "bitcoin", "investimento garantido",
        "prize", "you won", "refund", "wire transfer", "gift card",
    ),
    "anexo": (
        "em anexo", "veja o comprovante", "abra o documento", "baixe o arquivo",
        "nota fiscal", "comprovante de pagamento", "attached invoice", "download attached",
    ),
}

PT_STOPWORDS = [
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos",
    "e", "em", "essa", "esse", "esta", "este", "eu", "foi", "isso", "ja", "la", "lo",
    "mais", "mas", "me", "mesmo", "meu", "minha", "muito", "na", "nas", "no", "nos",
    "nao", "o", "os", "ou", "para", "pela", "pelo", "por", "que", "se", "sem",
    "ser", "seu", "sua", "sao", "tambem", "te", "tem", "um", "uma", "vez", "voce",
    "the", "and", "for", "you", "your", "with", "this", "that", "from", "have", "are",
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rebrand.ly", "goo.gl",
    "ow.ly", "shorturl.at", "encurtador.com.br", "rb.gy", "tiny.cc", "n9.cl", "l1nk.dev",
}

SUSPICIOUS_TLDS = {
    "zip", "mov", "top", "xyz", "icu", "click", "work", "tk", "ml", "cf", "ga", "gq",
    "buzz", "rest", "fit", "monster", "quest", "cyou", "sbs", "lol", "casa", "link",
}

FREEMAIL = {
    "gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "yahoo.com.br", "bol.com.br",
    "uol.com.br", "live.com", "icloud.com", "proton.me", "mail.ru", "yandex.com",
}

# marca -> dominio oficial (usado no teste de typosquatting e de marca "emprestada")
BRAND_DOMAINS: dict[str, str] = {
    "itau": "itau.com.br", "bradesco": "bradesco.com.br", "santander": "santander.com.br",
    "nubank": "nubank.com.br", "caixa": "caixa.gov.br", "bancodobrasil": "bb.com.br",
    "inter": "bancointer.com.br", "c6bank": "c6bank.com.br", "picpay": "picpay.com",
    "mercadopago": "mercadopago.com.br", "mercadolivre": "mercadolivre.com.br",
    "serasa": "serasa.com.br", "correios": "correios.com.br", "receita": "gov.br",
    "netflix": "netflix.com", "paypal": "paypal.com",
    "microsoft": "microsoft.com", "google": "google.com", "apple": "apple.com",
    "amazon": "amazon.com.br", "magalu": "magazineluiza.com.br",
    "americanas": "americanas.com.br", "whatsapp": "whatsapp.com",
    "instagram": "instagram.com", "steam": "steampowered.com", "binance": "binance.com",
    "spotify": "spotify.com", "ifood": "ifood.com.br", "uber": "uber.com",
}

URL_RE = re.compile("(?:https?://|www\\.)[^\\s<>\"\x27)]+", re.IGNORECASE)
IP_HOST_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?$")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
BAD_ATTACHMENTS = (".exe", ".scr", ".js", ".vbs", ".bat", ".cmd", ".jar", ".hta",
                   ".iso", ".img", ".zip", ".rar", ".7z", ".html", ".htm", ".lnk", ".msi")

NUMERIC_FEATURES = [
    "trig_urgencia", "trig_credencial", "trig_ameaca", "trig_dinheiro", "trig_anexo",
    "trig_total", "body_len", "subject_len", "caps_ratio", "exclamations", "digits_ratio",
    "n_links", "n_domains", "has_ip_url", "has_shortener", "has_suspicious_tld",
    "has_punycode", "url_has_at", "max_url_len", "brand_in_url_wrong_domain",
    "typosquat_min_dist", "typosquat_hit", "from_reply_mismatch",
    "display_brand_mismatch", "freemail_posing_brand", "spf_fail", "dkim_fail",
    "auth_missing", "domain_age_days", "domain_age_unknown", "domain_is_new",
    "suspicious_attachment",
]

TEXT_PREFIX = "txt:"


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def levenshtein(a: str, b: str) -> int:
    """Distancia de edicao em Python puro (sem dependencia externa)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text or "")


def host_of(url: str) -> str:
    host = re.sub(r"^https?://", "", url or "", flags=re.I)
    host = host.split("/")[0].split("?")[0].split("#")[0]
    if "@" in host:                      # http://usuario@dominio-malicioso.com
        host = host.split("@")[-1]
    return host.lower().strip(".")


@lru_cache(maxsize=8192)
def registrable(host: str) -> str:
    """Aproximacao de dominio registravel, suficiente para .com.br e afins."""
    parts = host.split(":")[0].split(".")
    if len(parts) <= 2:
        return ".".join(parts)
    if parts[-2] in {"com", "net", "org", "gov", "edu", "adv", "eng"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def domain_of_email(addr) -> str:
    addr = normalize_text(addr).strip().lower()
    match = EMAIL_RE.search(addr)
    if match:
        return match.group(0).split("@")[-1]
    return addr.split("@")[-1] if "@" in addr else ""


@lru_cache(maxsize=8192)
def typosquat(host: str) -> tuple[int, str]:
    """Menor distancia de edicao entre o rotulo do dominio e uma marca conhecida."""
    label = registrable(host).split(".")[0]
    label = re.sub(r"[^a-z0-9]", "", label.lower())
    if not label:
        return 99, ""
    best, brand = 99, ""
    for name in BRAND_DOMAINS:
        dist = levenshtein(label, name)
        if dist < best:
            best, brand = dist, name
    return best, brand


def brand_in_url_but_other_domain(urls: Iterable[str]) -> int:
    """itau.verificacao-cliente.com -> marca no subdominio, dominio de terceiro."""
    for url in urls:
        host = host_of(url)
        reg = registrable(host)
        blob = re.sub(r"[^a-z0-9]", "", url.lower())
        for brand, official in BRAND_DOMAINS.items():
            if brand in blob and not (reg == official or reg.endswith("." + official)):
                return 1
    return 0


@lru_cache(maxsize=512)
def whois_domain_age_days(domain: str) -> float:
    """Idade do dominio via WHOIS. Retorna -1 quando indisponivel (offline/erro)."""
    try:
        import datetime as _dt

        import whois  # type: ignore

        data = whois.whois(domain)
        created = data.creation_date
        if isinstance(created, list):
            created = created[0]
        if created is None:
            return -1.0
        if isinstance(created, str):
            created = _dt.datetime.fromisoformat(created[:19])
        return float((_dt.datetime.now() - created.replace(tzinfo=None)).days)
    except Exception:
        return -1.0


# --------------------------------------------------------------------------- #
# Extracao numerica por e-mail
# --------------------------------------------------------------------------- #
def full_text(row) -> str:
    subject = normalize_text(row.get("subject"))
    body = normalize_text(row.get("body"))
    return f"{subject}\n{subject}\n{body}"      # assunto pesa o dobro


def count_triggers(text: str) -> dict:
    low = text.lower()
    counts = {}
    total = 0
    for group, phrases in TRIGGER_GROUPS.items():
        hits = sum(low.count(phrase) for phrase in phrases)
        counts[f"trig_{group}"] = float(hits)
        total += hits
    counts["trig_total"] = float(total)
    return counts


def row_features(row, use_whois: bool = False) -> dict:
    subject = normalize_text(row.get("subject"))
    body = normalize_text(row.get("body"))
    text = f"{subject}\n{body}"
    feats = count_triggers(text)

    letters = [c for c in text if c.isalpha()]
    feats["body_len"] = float(len(body))
    feats["subject_len"] = float(len(subject))
    feats["caps_ratio"] = float(sum(c.isupper() for c in letters) / max(len(letters), 1))
    feats["exclamations"] = float(text.count("!"))
    feats["digits_ratio"] = float(sum(c.isdigit() for c in text) / max(len(text), 1))

    urls = extract_urls(text)
    extra = normalize_text(row.get("urls"))
    if extra:
        urls += [u for u in re.split(r"[\s,;|]+", extra) if u]
    hosts = [host_of(u) for u in urls if u]
    regs = {registrable(h) for h in hosts if h}

    feats["n_links"] = float(len(urls))
    feats["n_domains"] = float(len(regs))
    feats["has_ip_url"] = float(any(IP_HOST_RE.match(h) for h in hosts))
    feats["has_shortener"] = float(any(registrable(h) in SHORTENERS for h in hosts))
    feats["has_suspicious_tld"] = float(
        any(h.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS for h in hosts if "." in h)
    )
    feats["has_punycode"] = float(any(h.startswith("xn--") or not h.isascii() for h in hosts))
    feats["url_has_at"] = float(any("@" in u.split("//")[-1].split("/")[0] for u in urls))
    feats["max_url_len"] = float(max((len(u) for u in urls), default=0))
    feats["brand_in_url_wrong_domain"] = float(brand_in_url_but_other_domain(urls))

    dists = [typosquat(h)[0] for h in hosts if h]
    sender_domain = domain_of_email(row.get("from_addr"))
    if sender_domain:
        dists.append(typosquat(sender_domain)[0])
    min_dist = min(dists) if dists else 99
    feats["typosquat_min_dist"] = float(min(min_dist, 10))
    feats["typosquat_hit"] = float(1 <= min_dist <= 2)

    reply_domain = domain_of_email(row.get("reply_to"))
    feats["from_reply_mismatch"] = float(
        bool(reply_domain) and bool(sender_domain)
        and registrable(reply_domain) != registrable(sender_domain)
    )

    display = normalize_text(row.get("from_name")).lower()
    display_blob = re.sub(r"[^a-z0-9]", "", display)
    brand_claimed = next((b for b in BRAND_DOMAINS if b in display_blob), "")
    sender_reg = registrable(sender_domain) if sender_domain else ""
    official = BRAND_DOMAINS.get(brand_claimed, "")
    feats["display_brand_mismatch"] = float(
        bool(brand_claimed) and sender_reg not in ("", official)
        and not sender_reg.endswith("." + official)
    )
    feats["freemail_posing_brand"] = float(bool(brand_claimed) and sender_reg in FREEMAIL)

    spf = normalize_text(row.get("spf_result") or row.get("spf")).lower()
    dkim = normalize_text(row.get("dkim_result") or row.get("dkim")).lower()
    feats["spf_fail"] = float(any(k in spf for k in ("fail", "softfail", "none", "invalid")))
    feats["dkim_fail"] = float(any(k in dkim for k in ("fail", "none", "invalid")))
    feats["auth_missing"] = float(not spf and not dkim)

    age = row.get("domain_age_days")
    age = float(age) if age is not None and str(age) not in ("", "nan", "None") else -1.0
    if use_whois and age < 0 and sender_domain:
        age = whois_domain_age_days(registrable(sender_domain))
    feats["domain_age_days"] = age
    feats["domain_age_unknown"] = float(age < 0)
    feats["domain_is_new"] = float(0 <= age <= 60)

    attachments = normalize_text(row.get("attachments")).lower()
    feats["suspicious_attachment"] = float(any(ext in attachments for ext in BAD_ATTACHMENTS))
    return feats


# --------------------------------------------------------------------------- #
# Espaco de features (TF-IDF + numericas) reutilizavel no treino e na predicao
# --------------------------------------------------------------------------- #
class EmailFeatureSpace:
    """Mantem vetorizador e scaler juntos, para o mesmo espaco valer na predicao."""

    def __init__(self, max_features: int = 400, use_whois: bool = False):
        self.max_features = max_features
        self.use_whois = use_whois
        self.vectorizer: TfidfVectorizer | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []

    def _numeric_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = [row_features(row, self.use_whois) for _, row in df.iterrows()]
        frame = pd.DataFrame(rows, columns=NUMERIC_FEATURES)
        return frame.astype(float).fillna(0.0).reset_index(drop=True)

    def fit(self, df: pd.DataFrame) -> "EmailFeatureSpace":
        self.fit_transform(df)
        return self

    def fit_transform(self, df: pd.DataFrame):
        """Ajusta e ja devolve a matriz - evita extrair as features duas vezes."""
        texts = [full_text(row) for _, row in df.iterrows()]
        min_df = 2 if len(texts) > 40 else 1
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=min_df,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            stop_words=PT_STOPWORDS,
        ).fit(texts)
        numeric = self._numeric_frame(df)
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
            raise RuntimeError("EmailFeatureSpace.fit() precisa rodar antes de transform()")
        texts = [full_text(row) for _, row in df.iterrows()]
        return self._assemble(texts, self._numeric_frame(df))
