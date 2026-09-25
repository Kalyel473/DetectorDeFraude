"""Camada opcional de transformer (BERTimbau / DistilBERT) para texto.

Plugavel via `--model bert`. NAO e necessaria para o uso basico: os modulos de
e-mail e social rodam com TF-IDF + sklearn no modo --demo. Aqui o objetivo e
mostrar no video o trade-off entre performance e custo computacional.

Desbalanceamento em texto nao usa SMOTE (interpolar embeddings gera frase que
nao existe): usa-se peso de classe na funcao de perda.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

# BERTimbau (NeuralMind) - BERT treinado em portugues; alternativa multilingue menor.
DEFAULT_MODEL = "neuralmind/bert-base-portuguese-cased"
LIGHT_MODEL = "distilbert-base-multilingual-cased"


def availability() -> tuple[bool, str]:
    """(disponivel, mensagem). Nunca levanta excecao."""
    try:
        import torch  # type: ignore
        import transformers  # type: ignore

        device = "GPU (cuda)" if torch.cuda.is_available() else "CPU"
        return True, f"transformers {transformers.__version__} / torch {torch.__version__} em {device}"
    except Exception as exc:  # pragma: no cover
        return False, f"dependencias ausentes ({exc}). Instale: pip install transformers torch"


class BertTextClassifier:
    """Fine-tuning simples de classificacao binaria de texto."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_len: int = 192,
        epochs: int = 2,
        batch_size: int = 16,
        lr: float = 2e-5,
        seed: int = 42,
        fp16: bool | None = None,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.fp16 = fp16
        self.verbose = verbose
        self.tokenizer = None
        self.model = None
        self.device = None

    # ------------------------------------------------------------------ setup
    def _require(self):
        ok, message = availability()
        if not ok:
            raise RuntimeError(
                "Camada BERT indisponivel: " + message +
                "\nO baseline sklearn continua funcionando: use --model rf ou --model logreg."
            )
        import torch  # type: ignore

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.fp16 is None:
            self.fp16 = self.device == "cuda"
        return torch

    def _load_backbone(self, num_labels: int = 2, path: str | None = None):
        torch = self._require()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        source = path or self.model_name
        self.tokenizer = self._load_tokenizer(AutoTokenizer, source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=num_labels
        ).to(self.device)
        torch.manual_seed(self.seed)
        return torch

    @staticmethod
    def _load_tokenizer(auto_class, source: str):
        """Carrega o tokenizer com fallback.

        Checkpoints como o BERTimbau publicam apenas `vocab.txt`. Nas versoes
        novas de transformers o AutoTokenizer pede sentencepiece/tiktoken para
        converter isso, mas o BertTokenizerFast monta o WordPiece direto do
        vocabulario - fallback que mantem a camada BERT usavel sem dependencia
        extra.
        """
        try:
            return auto_class.from_pretrained(source)
        except Exception as first_error:
            try:
                from transformers import BertTokenizerFast  # type: ignore

                return BertTokenizerFast.from_pretrained(source)
            except Exception:
                raise RuntimeError(
                    f"Nao foi possivel carregar o tokenizer de {source}: {first_error}\n"
                    "Saidas possiveis: pip install sentencepiece   ou   escolher um "
                    "checkpoint com tokenizer.json (ex.: --bert-model "
                    "distilbert-base-multilingual-cased)."
                ) from first_error

    def _batches(self, texts, labels, torch, shuffle: bool):
        order = np.arange(len(texts))
        if shuffle:
            np.random.default_rng(self.seed).shuffle(order)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = self.tokenizer(
                [str(texts[i]) for i in idx],
                truncation=True,
                padding=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            target = None
            if labels is not None:
                target = torch.tensor([int(labels[i]) for i in idx], device=self.device)
            yield batch, target

    # -------------------------------------------------------------------- fit
    def fit(self, texts, labels) -> "BertTextClassifier":
        torch = self._load_backbone()
        labels = np.asarray(labels).astype(int)
        counts = np.bincount(labels, minlength=2).astype(float)
        weights = torch.tensor(
            (counts.sum() / np.maximum(counts, 1.0)) / 2.0, dtype=torch.float32, device=self.device
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        steps = max(1, (len(texts) // self.batch_size + 1) * self.epochs)
        try:
            from transformers import get_linear_schedule_with_warmup  # type: ignore

            scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
        except Exception:
            scheduler = None
        scaler = torch.amp.GradScaler("cuda", enabled=bool(self.fp16))

        self.model.train()
        for epoch in range(self.epochs):
            total, seen = 0.0, 0
            for batch, target in self._batches(texts, labels, torch, shuffle=True):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=bool(self.fp16)):
                    logits = self.model(**batch).logits
                    loss = loss_fn(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                total += float(loss.item()) * len(target)
                seen += len(target)
            if self.verbose:
                print(f"  [bert] epoca {epoch + 1}/{self.epochs} - loss {total / max(seen, 1):.4f}")
        return self

    # ---------------------------------------------------------------- predict
    def predict_proba(self, texts) -> np.ndarray:
        torch = self._require()
        if self.model is None:
            raise RuntimeError("Modelo BERT nao carregado (use fit() ou load()).")
        self.model.eval()
        out = []
        with torch.no_grad():
            for batch, _ in self._batches(list(texts), None, torch, shuffle=False):
                with torch.amp.autocast("cuda", enabled=bool(self.fp16)):
                    logits = self.model(**batch).logits.float()
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 2))

    def embed(self, texts) -> np.ndarray:
        """Embedding do token [CLS] - permite reuso no modulo de redes sociais."""
        torch = self._require()
        self.model.eval()
        out = []
        with torch.no_grad():
            for batch, _ in self._batches(list(texts), None, torch, shuffle=False):
                hidden = self.model.base_model(**batch).last_hidden_state[:, 0, :]
                out.append(hidden.float().cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 1))

    # ------------------------------------------------------------- disco
    def save(self, path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        meta = {
            "model_name": self.model_name,
            "max_len": self.max_len,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }
        (path / "fraudshield.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return str(path)

    @classmethod
    def load(cls, path) -> "BertTextClassifier":
        path = Path(path)
        meta = {}
        meta_file = path / "fraudshield.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        obj = cls(
            model_name=meta.get("model_name", DEFAULT_MODEL),
            max_len=int(meta.get("max_len", 192)),
            epochs=int(meta.get("epochs", 2)),
            batch_size=int(meta.get("batch_size", 16)),
            lr=float(meta.get("lr", 2e-5)),
        )
        obj._load_backbone(path=str(path))
        return obj


# --------------------------------------------------------------------------- #
# Explicabilidade para texto: oclusao de palavras
# --------------------------------------------------------------------------- #
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9._-]{3,}")


def explain_by_occlusion(clf: BertTextClassifier, text: str, top_k: int = 5,
                         max_words: int = 40) -> list[tuple[str, float]]:
    """Remove uma palavra por vez e mede a queda no score de fraude.

    Equivalente conceitual ao SHAP para texto, sem o custo de amostragem:
    mostra quais termos sustentam a decisao do transformer.
    """
    text = str(text or "")
    seen, candidates = set(), []
    for match in WORD_RE.finditer(text):
        word = match.group(0)
        key = word.lower()
        if key not in seen:
            seen.add(key)
            candidates.append(word)
        if len(candidates) >= max_words:
            break
    if not candidates:
        return []

    base = float(clf.predict_proba([text])[0, 1])
    variants = [re.sub(re.escape(word), " ", text, flags=re.IGNORECASE) for word in candidates]
    probs = clf.predict_proba(variants)[:, 1]
    deltas = [(word, base - float(p)) for word, p in zip(candidates, probs)]
    deltas.sort(key=lambda item: item[1], reverse=True)
    return [(w, d) for w, d in deltas if d > 0][:top_k]
