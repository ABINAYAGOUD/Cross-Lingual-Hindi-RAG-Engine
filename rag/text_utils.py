"""Devanagari-aware text normalisation and sentence segmentation."""
import re
import unicodedata

# Devanagari digits -> ASCII, so "१९९७" and "1997" match in both the document and queries.
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# Zero-width / invisible characters that PDFs often inject and that break matching.
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)

# Sentence boundaries: Hindi danda (।), double danda (॥), ? and !, plus an English
# full stop only after 2+ lowercase letters/digits (so "A.P.J." and "Dr." don't split).
_SENT_SPLIT = re.compile(r"(?<=[।॥?!])\s+|(?<=[a-z0-9\)]{2}\.)\s+")

# A Devanagari combining sign (vowel sign, nukta, virama, anusvara...) can never legally repeat
# back-to-back. Some PDF text layers emit them twice ("सम्माान", "अब्दुुल"), so collapse repeats.
_SIGNS = "\u0900-\u0903\u093A-\u093C\u093E-\u094F\u0955-\u0957\u0962\u0963"
_DUP_SIGN = re.compile(rf"([{_SIGNS}]{{1,2}})\1+")   # also catches doubled pairs like "ांां"

_TOKEN = re.compile(r"[\u0900-\u0963\u0970-\u097F]+|[A-Za-z0-9]+")


def normalize(text: str) -> str:
    """NFC + strip invisible chars + ASCII digits + tidy whitespace (UTF-8 safe)."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE)
    text = text.replace("\u00a0", " ")
    text = text.translate(_DEV_DIGITS)
    text = _DUP_SIGN.sub(r"\1", text)
    text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)  # de-hyphenate English line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def tokenize(text: str) -> list[str]:
    """Word tokens for BM25. (\\w would split Hindi words at vowel signs, so we use
    explicit Unicode ranges, excluding the danda U+0964/U+0965.)"""
    return _TOKEN.findall(normalize(text).lower())


def devanagari_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha() or "\u0900" <= c <= "\u097F"]
    if not letters:
        return 0.0
    return sum("\u0900" <= c <= "\u097F" for c in letters) / len(letters)
