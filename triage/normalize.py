"""Text normalization shared by the rules and the models.

Folding makes typed Turkish, ASCII typing ("agrim") and later speech transcripts look alike.
"""

import re

_FOLD = str.maketrans("çğıöşüâîûÇĞİIÖŞÜÂÎÛ", "cgiosuaiuCGIIOSUAIU")
_REPEATS = re.compile(r"(\w)\1{2,}")  # three or more of one letter: "gelinnn"
_NON_WORD = re.compile(r"[^\w\s]")


def fold(text: str, keep: str = "") -> str:
    """Lowercase, fold Turkish characters, drop punctuation (except `keep`), squeeze spaces."""
    # Fold before lowercasing: Python lowercases "I" to "i" and "İ" to "i̇" (with a dot).
    text = text.translate(_FOLD).lower()
    text = _REPEATS.sub(r"\1", text)
    if keep:
        text = re.sub(rf"[^\w\s{re.escape(keep)}]", " ", text)
    else:
        text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())
