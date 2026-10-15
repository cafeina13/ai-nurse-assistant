from triage.pipeline import Settings, Triage


class FixedModel:
    """A stand-in model that always predicts one cell with a given confidence."""

    def __init__(self, cell: str, confidence: float):
        self.name = "fixed"
        self.cell, self.confidence = cell, confidence

    def predict_proba(self, texts):
        rest = (1 - self.confidence) / 27
        return [{self.cell: self.confidence, "unclear/soon": rest} for _ in texts]


def run(text, cell="comfort_supplies/routine", confidence=0.95, **kw):
    return Triage(FixedModel(cell, confidence)).run(text, **kw)


def test_buttons_skip_the_model():
    result = run("[Su]", cell="pain/critical", channel="button")
    assert result.cell == "comfort_supplies/routine"
    assert result.confidence is None and not result.interrupt


def test_rule_wins_over_the_model():
    result = run("Nefes alamıyorum", cell="comfort_supplies/routine", confidence=0.99)
    assert result.cell == "breathing/critical"
    assert any(r.startswith("rule:") for r in result.reasons)
    # A rule hit the model disagrees with tops the queue but does not interrupt.
    assert not result.interrupt


def test_rule_and_model_agree_interrupts():
    result = run("Nefes alamıyorum", cell="breathing/critical", confidence=0.6)
    assert result.cell == "breathing/critical" and result.interrupt


def test_confident_model_alone_interrupts():
    assert run("bir şey oldu çabuk", cell="unclear/critical", confidence=0.95).interrupt
    assert not run("bir şey oldu çabuk", cell="unclear/critical", confidence=0.7).interrupt


def test_low_confidence_is_flagged_uncertain():
    assert run("hmm", cell="information/routine", confidence=0.3).uncertain
    assert not run("hmm", cell="information/routine", confidence=0.8).uncertain


def test_patient_context_raises_toilet_help():
    assisted = run("tuvalete gitmem lazım", cell="physical_help/routine", mobility="needs_assistance")
    independent = run("tuvalete gitmem lazım", cell="physical_help/routine", mobility="independent")
    assert assisted.cell == "physical_help/soon"
    assert independent.cell == "physical_help/routine"
    assert run("[Tuvalet]", channel="button", mobility="bedbound").cell == "physical_help/soon"


def test_results_carry_a_version():
    assert run("su").version.startswith("rules:")


def test_settings_change_thresholds():
    strict = Triage(FixedModel("unclear/critical", 0.8), settings=Settings(interrupt_above=0.75))
    assert strict.run("bir şey oldu").interrupt
