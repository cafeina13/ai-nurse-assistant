from triage.models import EnsembleModel, make_model

TEXTS = ["nefes alamıyorum", "nefesim daralıyor çok kötü", "su alabilir miyim", "bir bardak su lütfen",
         "başım çok ağrıyor", "ağrım var ilaç lazım"] * 3
CELLS = ["breathing/critical", "breathing/critical", "comfort_supplies/routine", "comfort_supplies/routine",
         "pain/soon", "pain/soon"] * 3


class Fixed:
    """A member that always gives the same probabilities."""

    def __init__(self, probs):
        self.probs, self.name, self.settings = probs, "fixed", {}

    def fit(self, texts, cells):
        return self

    def predict_proba(self, texts):
        return [dict(self.probs) for _ in texts]


def test_ensemble_averages_members():
    ens = EnsembleModel([Fixed({"a/critical": 0.8, "b/soon": 0.2}), Fixed({"b/soon": 0.6, "c/routine": 0.4})])
    (probs,) = ens.predict_proba(["x"])
    assert probs == {"a/critical": 0.4, "b/soon": 0.4, "c/routine": 0.2}


def test_ensemble_spec_builds_and_predicts():
    model = make_model("ens:tfidf+tfidf").fit(TEXTS, CELLS)
    assert model.name.startswith("ens(tfidf")
    (probs,) = model.predict_proba(["nefes alamıyorum"])
    assert max(probs, key=probs.get) == "breathing/critical"
    assert abs(sum(probs.values()) - 1) < 1e-9


def test_hybrid_wraps_an_ensemble():
    model = make_model("hybrid-critical:ens:tfidf+tfidf")
    assert model.name.startswith("hybrid-critical(ens(")
