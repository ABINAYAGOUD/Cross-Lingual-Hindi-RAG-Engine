import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag.text_utils import normalize, split_sentences, tokenize, devanagari_ratio
from rag.chunking import pack_sentences, table_to_chunks, is_question_list


def test_nfc_and_digits():
    # decomposed nukta (क + ़) must equal composed form after NFC; Devanagari digits -> ASCII
    assert normalize("क\u093c") == normalize("क\u093c")
    assert normalize("वर्ष १९९७\u200b") == "वर्ष 1997"


def test_danda_split_and_abbrev():
    s = split_sentences("कलाम का जन्म 1931 में हुआ। वे वैज्ञानिक थे। A.P.J. Abdul Kalam was born in India. He died in 2015.")
    assert s[0].endswith("हुआ।") and s[1].endswith("थे।")
    assert any(x.startswith("A.P.J. Abdul Kalam") for x in s)   # initials not split
    assert s[-1] == "He died in 2015."


def test_tokenize_keeps_matras_and_drops_danda():
    assert tokenize("भारत रत्न, 1997।") == ["भारत", "रत्न", "1997"]


def test_chunks_respect_size_and_overlap():
    sents = [f"यह वाक्य संख्या {i} है।" for i in range(40)]
    chunks = pack_sentences(sents, max_chars=120, overlap_chars=30)
    assert all(len(c) <= 140 for c in chunks)
    assert len(chunks) > 3
    # consecutive chunks share at least one sentence (overlap)
    assert any(set(a.split("।")) & set(b.split("।")) for a, b in zip(chunks, chunks[1:]))
    # every sentence is covered
    assert all(any(s in c for c in chunks) for s in sents)


def test_long_sentence_hard_split():
    chunks = pack_sentences(["शब्द " * 300], 200, 40)
    assert all(len(c) <= 205 for c in chunks)


def test_table_rows_stay_whole():
    rows = [["वर्ष", "घटना"], ["1980", "रोहिणी उपग्रह"], ["1997", "भारत रत्न"]]
    out = table_to_chunks(rows, 600)
    assert out == ["वर्ष: 1980 | घटना: रोहिणी उपग्रह", "वर्ष: 1997 | घटना: भारत रत्न"]
    assert table_to_chunks(rows, 600, per_row=False) == ["वर्ष: 1980 | घटना: रोहिणी उपग्रह\nवर्ष: 1997 | घटना: भारत रत्न"]


def test_devanagari_ratio():
    assert devanagari_ratio("हिंदी text") > 0.4


def test_duplicate_matras_collapsed():
    assert normalize("सम्माान अब्दुुल क्याा जैनुलाब्दीीन स्पेेस") == "सम्मान अब्दुल क्या जैनुलाब्दीन स्पेस"
    assert normalize("कलाम") == "कलाम"          # normal text untouched


def test_doubled_pair_and_question_list():
    assert normalize("मूल्यांांकन") == "मूल्यांकन"
    assert is_question_list("1. क्या है? 2. कौन था? 3. कब हुआ?")
    assert not is_question_list("कलाम को 1997 में भारत रत्न मिला। क्या यह सच है?")


def test_parse_sources_variants():
    from rag.generate import parse_sources
    cases = [
        ("1997 में प्रदान। SOURCES: 1,2,5", ("1997 में प्रदान।", [1, 2, 5])),
        ("Answer text.\n\nSOURCES: 3", ("Answer text.", [3])),
        ("Answer. **Sources:** 1, 4", ("Answer.", [1, 4])),
        ("Answer. SOURCES：2", ("Answer.", [2])),
        ("Answer. (Sources: [1], [9])", ("Answer. (", [1])),   # 9 > n_hits is dropped
        ("No marker at all", ("No marker at all", [])),
        ("The sources: government records say 1997", ("The sources: government records say 1997", [])),
    ]
    for raw, (ans, nums) in cases[:4] + cases[5:]:
        assert parse_sources(raw, 5) == (ans, nums), raw
    assert parse_sources("Answer. SOURCES: 1,2,5", 5)[0] == "Answer."


def test_answer_language_is_decided_in_code():
    from rag.generate import answer_language, build_prompt
    from rag.store import Hit
    assert answer_language("Who co-wrote the autobiography, and in what year was it published?") == "English"
    assert answer_language("How and where did Kalam die in 2015?") == "English"
    assert answer_language("कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?") == "Hindi"
    assert answer_language("SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?") == "Hindi"
    hit = Hit(1, "पाठ", 3, "खंड", "text", 0.8)
    assert "ENGLISH" in build_prompt("Who wrote it?", [hit])
    assert "HINDI" in build_prompt("किसने लिखा?", [hit])