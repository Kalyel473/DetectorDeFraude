"""Modulo 2 - phishing em e-mail.

Datasets de referencia (publicos, usados em pesquisa academica):
  * legitimos  : Enron Email Dataset
  * fraudulentos: Nazario Phishing Corpus / feeds do PhishTank
Basta gerar um CSV com as colunas from_name, from_addr, reply_to, subject,
body, spf_result, dkim_result, attachments, label (1 = phishing) e apontar
com --data. O modo --demo usa a amostra sintetica do repositorio.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..config import RAW_DIR, SAMPLES_DIR
from ..features.text_features import (
    TEXT_PREFIX,
    EmailFeatureSpace,
    domain_of_email,
    extract_urls,
    full_text,
    host_of,
    typosquat,
)
from .base import FraudModule, read_table, text_term_reason

SAMPLE_NAME = "emails_amostra.csv"
RAW_CANDIDATES = ("emails.csv", "phishing.csv", "enron_nazario.csv")
LABEL_CANDIDATES = ("label", "class", "is_phishing", "phishing", "fraude")


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<br\s*/?>|</p>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
    return re.sub(r"[ \t]{2,}", " ", text)


def parse_eml(path: str | Path) -> pd.DataFrame:
    """Le um .eml/.msg-texto real e monta a linha que o modelo espera."""
    from email import policy
    from email.parser import BytesParser
    from email.utils import parseaddr

    path = Path(path)
    with path.open("rb") as handle:
        message = BytesParser(policy=policy.default).parse(handle)

    from_name, from_addr = parseaddr(message.get("From", ""))
    reply_to = parseaddr(message.get("Reply-To", ""))[1]
    subject = str(message.get("Subject", ""))

    body_parts, attachments = [], []
    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            filename = part.get_filename()
            if filename and "attachment" in disposition.lower():
                attachments.append(filename)
                continue
            if ctype == "text/plain":
                body_parts.append(part.get_content())
            elif ctype == "text/html":
                body_parts.append(_html_to_text(part.get_content()))
    else:
        content = message.get_content()
        if message.get_content_type() == "text/html":
            content = _html_to_text(content)
        body_parts.append(content)

    body = "\n".join(str(p) for p in body_parts if p)
    auth = " ".join(
        str(message.get(header, ""))
        for header in ("Authentication-Results", "Received-SPF", "ARC-Authentication-Results")
    ).lower()
    spf_match = re.search(r"spf=(\w+)", auth) or re.search(r"^(pass|fail|softfail|none)", auth)
    dkim_match = re.search(r"dkim=(\w+)", auth)

    return pd.DataFrame([{
        "message_id": message.get("Message-ID", path.name),
        "from_name": from_name,
        "from_addr": from_addr,
        "reply_to": reply_to,
        "subject": subject,
        "body": body,
        "spf_result": spf_match.group(1) if spf_match else "",
        "dkim_result": dkim_match.group(1) if dkim_match else "",
        "attachments": ";".join(attachments),
        "source_file": str(path),
    }])


class EmailPhishingModule(FraudModule):
    key = "email"
    title = "Phishing em e-mail"
    description = "TF-IDF + palavras-gatilho + cabecalho (SPF/DKIM/Reply-To) + analise de URL"
    sample_file = SAMPLE_NAME
    default_model = "rf"
    positive_name = "PHISHING"
    negative_name = "LEGITIMO"
    dataset_hint = (
        "Combine Enron (legitimos) com Nazario/PhishTank (phishing) em data/raw/emails.csv, "
        "ou use --demo para a amostra do repositorio."
    )

    def __init__(self, use_whois: bool = False, max_text_features: int = 400):
        self.use_whois = use_whois
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
            space = EmailFeatureSpace(
                max_features=self.max_text_features, use_whois=self.use_whois
            )
            artifacts["space"] = space
            X, raw = space.fit_transform(df)
        else:
            space = artifacts["space"]
            space.use_whois = self.use_whois or space.use_whois
            X, raw = space.transform(df)
        label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
        y = df[label_col].astype(int).to_numpy() if label_col else None
        artifacts["feature_names"] = space.feature_names
        artifacts["label_col"] = label_col
        return X, y, space.feature_names, artifacts, raw

    def texts(self, df: pd.DataFrame) -> list[str]:
        return [full_text(row) for _, row in df.reset_index(drop=True).iterrows()]

    def parse_input(self, path: str) -> pd.DataFrame:
        if str(path).lower().endswith((".eml", ".msg", ".email")):
            return parse_eml(path)
        return read_table(path)

    # ------------------------------------------------------------------ saida
    def describe(self, position: int, row: pd.Series) -> str:
        sender = str(row.get("from_addr", "") or "").strip()
        return f"e-mail #{position} de {sender}" if sender else f"e-mail #{position}"

    def humanize(self, feature: str, value: float, contribution: float, row: pd.Series) -> str:
        value = float(value)
        if feature.startswith(TEXT_PREFIX):
            return text_term_reason(feature, value)

        if feature.startswith("trig_"):
            group = feature[5:]
            names = {
                "urgencia": "urgencia/pressao de tempo",
                "credencial": "pedido de credencial ou clique",
                "ameaca": "ameaca de bloqueio/penalidade",
                "dinheiro": "isca financeira (premio, pix, boleto)",
                "anexo": "insistencia em anexo/documento",
                "total": "palavras-gatilho no total",
            }
            return f"{value:.0f} termo(s) de {names.get(group, group)}"
        if feature == "spf_fail":
            return "SPF invalido (remetente nao autorizado pelo dominio)" if value else "SPF valido"
        if feature == "dkim_fail":
            return "DKIM invalido ou ausente (assinatura nao confere)" if value else "DKIM valido"
        if feature == "auth_missing":
            return "cabecalho sem nenhuma autenticacao (SPF/DKIM ausentes)"
        if feature == "from_reply_mismatch":
            reply = domain_of_email(row.get("reply_to")) if hasattr(row, "get") else ""
            extra = f" ({reply})" if reply else ""
            return f"Reply-To aponta para dominio diferente do From{extra}"
        if feature == "display_brand_mismatch":
            return "nome exibido usa uma marca, mas o dominio real e outro"
        if feature == "freemail_posing_brand":
            return "marca no nome exibido com remetente em provedor gratuito"
        if feature == "typosquat_hit" or feature == "typosquat_min_dist":
            brand = ""
            if hasattr(row, "get"):
                domain = domain_of_email(row.get("from_addr"))
                if domain:
                    brand = typosquat(domain)[1]
            suffix = f" ({brand})" if brand else ""
            return f"dominio a {value:.0f} edicao(oes) de distancia de uma marca conhecida{suffix} - typosquatting"
        if feature == "brand_in_url_wrong_domain":
            return "marca conhecida usada na URL, mas o dominio registrado e de terceiro"
        if feature == "has_ip_url":
            return "URL aponta para IP puro em vez de dominio"
        if feature == "has_shortener":
            return "URL encurtada esconde o destino real"
        if feature == "has_suspicious_tld":
            return "TLD de alto abuso na URL (.top/.zip/.xyz e similares)"
        if feature == "has_punycode":
            return "dominio com punycode/caracteres nao-ASCII (homoglifos)"
        if feature == "url_has_at":
            return "URL com @ antes do host (tecnica de ofuscacao)"
        if feature == "domain_age_days":
            if value < 0:
                return "idade do dominio do remetente desconhecida"
            return f"dominio do remetente criado ha {value:.0f} dias"
        if feature == "domain_is_new":
            return "dominio do remetente registrado recentemente (<= 60 dias)"
        if feature == "domain_age_unknown":
            return "WHOIS do dominio do remetente indisponivel"
        if feature == "suspicious_attachment":
            return "anexo de extensao perigosa (.zip/.exe/.html e similares)"
        if feature == "caps_ratio":
            return f"{value * 100:.0f}% do texto em CAIXA ALTA"
        if feature == "exclamations":
            return f"{value:.0f} pontos de exclamacao"
        if feature == "n_links":
            return f"{value:.0f} link(s) no corpo do e-mail"
        if feature == "n_domains":
            return f"{value:.0f} dominio(s) distinto(s) nos links"
        if feature == "max_url_len":
            return f"URL mais longa com {value:.0f} caracteres"
        if feature == "body_len":
            return f"corpo com {value:.0f} caracteres"
        if feature == "subject_len":
            return f"assunto com {value:.0f} caracteres"
        if feature == "digits_ratio":
            return f"{value * 100:.1f}% do texto sao digitos"
        return f"{feature} = {value:.4g}"

    def extra_evidence(self, row: pd.Series) -> list[str]:
        """Itens objetivos mostrados junto da explicacao do modelo."""
        evidence = []
        subject = str(row.get("subject", "") or "").strip()
        if subject:
            evidence.append(f'assunto: "{subject[:70]}"')
        urls = extract_urls(full_text(row))
        extra = str(row.get("urls", "") or "")
        if extra:
            urls += [u for u in re.split(r"[\s,;|]+", extra) if u]
        hosts = sorted({host_of(u) for u in urls if u})
        if hosts:
            evidence.append("dominios nos links: " + ", ".join(hosts[:4]))
        sender = domain_of_email(row.get("from_addr"))
        if sender:
            distance, brand = typosquat(sender)
            if 1 <= distance <= 2:
                evidence.append(f"remetente {sender} parece imitar {brand} (Levenshtein={distance})")
        return evidence
