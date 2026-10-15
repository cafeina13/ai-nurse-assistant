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
