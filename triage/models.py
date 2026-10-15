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


def make_model(spec: str, C: float = 10.0):
    """"tfidf" or "emb:<hugging face id>"."""
    if spec == "tfidf":
        return TfidfModel(C=C)
    if spec.startswith("emb:"):
        return EmbeddingModel(spec[4:], C=C)
    raise ValueError(f"unknown model {spec!r}")
