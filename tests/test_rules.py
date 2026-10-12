import pytest

from triage.normalize import fold
from triage.rules import compile_pattern, find_hits, load_rules

RULES = load_rules()


def cells(text):
    return [h.cell for h in find_hits(text, RULES)]


def test_fold():
    assert fold("Göğsüm SIKIŞIYOR!!") == "gogsum sikisiyor"
    assert fold("İlacımı") == "ilacimi"
    assert fold("gelinnn çabuuuk") == "gelin cabuk"


def test_pattern_syntax():
    # A leading group is a required choice; a group after a stem is an optional suffix.
    assert compile_pattern("(birden/aniden) baş").search("aniden basim")
    assert not compile_pattern("(birden/aniden) baş").search("basim")
    assert compile_pattern("düştü(m)").search("dustu")
    # "..." allows up to four words in between.
    assert compile_pattern("göğs ... ağrı").search("gogsume bi agri saplandi")
    assert not compile_pattern("göğs ... ağrı").search("gogsum bir iki uc dort bes agri")


@pytest.mark.parametrize("text, cell", [
    ("Nefes alamıyorum", "breathing/critical"),
    ("nefes alamiyom", "breathing/critical"),
    ("BABAM BOĞULUYOR", "breathing/critical"),
    ("göğsüme bi ağrı saplandı", "pain/critical"),
    ("Annem uyanmıyor", "neuro/critical"),
    ("yüzü kaydı, konuşamıyor", "neuro/critical"),
    ("Babam düştü", "fall_injury/critical"),
    ("yatak kan içinde kaldı", "fall_injury/critical"),
    ("dudaklarım şişti", "medication/critical"),
    ("ölmek istiyorum", "emotional/critical"),
    ("yaşamak istemiyorum", "emotional/critical"),
])
def test_red_flags_fire(text, cell):
    assert cell in cells(text)


@pytest.mark.parametrize("text", [
    "Su alabilir miyim",
    "Ateşim düştü",  # good news, not a fall
    "maskem düştü",  # an object, not a person
    "kan sonuçlarım sisteme düştü mü",
    "Oksijen maskesi yüzümden kaydı",  # the mask slid, not the face
    "bilinci yerinde, uyuyor",
])
def test_known_traps_do_not_fire(text):
    assert cells(text) == []


def test_rules_only_target_critical_cells():
    assert all(rule.cell.endswith("/critical") for rule in RULES)
