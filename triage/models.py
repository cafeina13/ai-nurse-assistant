"""Cell classifiers. Each predicts one grid cell ("category/urgency") per text."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from triage.normalize import fold


class TfidfModel:
    """Baseline: TF-IDF over character pieces and word pairs, then logistic regression.

    Character pieces survive Turkish suffixes ("ağrı-m-ı-yor") and ASCII typing ("agrim"),
    which word matching does not.
    """

    def __init__(self, C: float = 10.0, seed: int = 0):
        self.name = f"tfidf-char2-5+word1-2-lr-C{C}"
        self.settings = {"C": C, "seed": seed, "char_ngrams": [2, 5], "word_ngrams": [1, 2]}
        self.pipeline = make_pipeline(
            make_union(
                TfidfVectorizer(preprocessor=fold, analyzer="char_wb", ngram_range=(2, 5),
                                sublinear_tf=True, min_df=2),
                TfidfVectorizer(preprocessor=fold, analyzer="word", ngram_range=(1, 2),
                                sublinear_tf=True, min_df=2),
            ),
            LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed),
        )

    def fit(self, texts: list[str], cells: list[str]) -> "TfidfModel":
        self.pipeline.fit(texts, cells)
        return self

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        # Plain str and float, not numpy types: results go into the JSON event log.
        classes = [str(c) for c in self.pipeline.classes_]
        return [dict(zip(classes, map(float, row))) for row in self.pipeline.predict_proba(texts)]


# Some embedding models are trained to expect a prefix on every input.
EMBEDDING_PREFIX = {"intfloat/multilingual-e5-base": "query: ", "intfloat/multilingual-e5-large": "query: "}


class EmbeddingModel:
    """Sentence embeddings (what a message means) under the same logistic regression as the
    baseline, so a comparison isolates the text representation."""

    def __init__(self, model_id: str, C: float = 10.0, seed: int = 0, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.name = f"emb-{model_id.split('/')[-1]}-lr-C{C}"
        self.settings = {"embedding": model_id, "C": C, "seed": seed}
        self.encoder = SentenceTransformer(model_id, device=device)
        self.prefix = EMBEDDING_PREFIX.get(model_id, "")
        self.classifier = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed)
        self._cache: dict[str, list[float]] = {}

    def _encode(self, texts: list[str]):
        import numpy as np

        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = self.encoder.encode([self.prefix + t for t in missing], batch_size=64,
                                          normalize_embeddings=True, show_progress_bar=False)
            self._cache.update(zip(missing, vectors))
        return np.array([self._cache[t] for t in texts])

    def fit(self, texts: list[str], cells: list[str]) -> "EmbeddingModel":
        self.classifier.fit(self._encode(texts), cells)
        return self

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        if not texts:
            return []
        classes = [str(c) for c in self.classifier.classes_]
        return [dict(zip(classes, map(float, row))) for row in self.classifier.predict_proba(self._encode(texts))]


class BertModel:
    """A Turkish BERT fine-tuned end to end on the cells (needs a GPU; run it on Kaggle).

    Standard settings, fixed before any run and not tuned on a test set: 4 epochs, lr 3e-5
    with 10% warm-up, batch 32, weight decay 0.01, texts cut at 64 tokens (requests are short),
    and class weights like the baseline's class_weight="balanced". Raw text goes in, not folded:
    a cased model can use the Turkish letters when they are there.
    """

    def __init__(self, model_id: str, epochs: int = 4, lr: float = 3e-5, batch_size: int = 32,
                 max_length: int = 64, seed: int = 0, device: str | None = None):
        self.model_id = model_id
        self.name = f"bert-{model_id.split('/')[-1]}-e{epochs}-lr{lr}"
        self.settings = {"model": model_id, "epochs": epochs, "lr": lr, "batch_size": batch_size,
                         "max_length": max_length, "seed": seed, "warmup": 0.1, "weight_decay": 0.01,
                         "class_weights": "balanced"}
        self.epochs, self.lr, self.batch_size, self.max_length, self.seed = epochs, lr, batch_size, max_length, seed
        self.device = device

    def fit(self, texts: list[str], cells: list[str]) -> "BertModel":
        import os

        import numpy as np
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  get_linear_schedule_with_warmup)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = sorted(set(cells))
        index = {c: i for i, c in enumerate(self.classes)}
        labels = torch.tensor([index[c] for c in cells])
        counts = torch.bincount(labels, minlength=len(self.classes)).float()
        weights = (len(labels) / (len(self.classes) * counts)).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, num_labels=len(self.classes)).to(self.device)
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        steps = self.epochs * -(-len(texts) // self.batch_size)
        no_decay = ("bias", "LayerNorm.weight")
        params = [
            {"params": [p for n, p in self.model.named_parameters() if not any(k in n for k in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in self.model.named_parameters() if any(k in n for k in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler(enabled=self.device == "cuda")
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        generator = torch.Generator().manual_seed(self.seed)
        self.model.train()
        for epoch in range(self.epochs):
            order = torch.randperm(len(texts), generator=generator)
            total = 0.0
            for start in range(0, len(texts), self.batch_size):
                batch = order[start:start + self.batch_size]
                inputs = {k: v[batch].to(self.device) for k, v in enc.items()}
                with torch.autocast(self.device, enabled=self.device == "cuda"):
                    logits = self.model(**inputs).logits
                loss = loss_fn(logits.float(), labels[batch].to(self.device))
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                schedule.step()
                total += loss.item() * len(batch)
            print(f"{self.name} epoch {epoch + 1}/{self.epochs} loss {total / len(texts):.4f}", flush=True)
        self.model.eval()
        if os.environ.get("TRIAGE_SAVE_DIR"):  # keep the weights, e.g. for the demo later
            out = os.path.join(os.environ["TRIAGE_SAVE_DIR"], self.name)
            self.model.save_pretrained(out)
            self.tokenizer.save_pretrained(out)
        return self

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        import torch

        out = []
        with torch.no_grad():
            for start in range(0, len(texts), 128):
                enc = self.tokenizer(texts[start:start + 128], truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors="pt").to(self.device)
                with torch.autocast(self.device, enabled=self.device == "cuda"):
                    probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().tolist()
                out += [dict(zip(self.classes, map(float, row))) for row in probs]
        return out


def make_model(spec: str, C: float = 10.0):
    """"tfidf", "emb:<hugging face id>", "bert:<hugging face id>", "protocol" or
    "hybrid-<mode>:<classifier spec>"."""
    if spec == "tfidf":
        return TfidfModel(C=C)
    if spec.startswith("emb:"):
        return EmbeddingModel(spec[4:], C=C)
    if spec.startswith("bert:"):
        return BertModel(spec[5:])
    if spec == "protocol":
        from triage.protocol import ProtocolModel
        return ProtocolModel()
    if spec.startswith("hybrid-"):  # hybrid-<mode>:<classifier spec>, e.g. hybrid-raise:tfidf
        from triage.protocol import HybridModel
        mode, inner = spec[len("hybrid-"):].split(":", 1)
        return HybridModel(make_model(inner, C=C), mode=mode)
    raise ValueError(f"unknown model {spec!r}")
