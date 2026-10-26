"""Grounded answer generation. Providers: anthropic | openai | ollama | none."""
from __future__ import annotations

import re

import requests

from . import config
from .store import Hit
from .text_utils import devanagari_ratio

SYSTEM = (
    "You answer questions about a Hindi document using ONLY the numbered context passages "
    "provided. Do not use outside knowledge and do not add details the passages do not state. "
    "Answer in 1-3 short sentences, in the language requested at the end of the user message "
    "(the passages are Hindi, but that does not decide the answer language). "
    "Be complete on every specific the question asks for: for 'how' say what happened; for "
    "'where' give the institution AND the place; for 'who' and 'when' give the names and "
    "dates stated. Attribute each fact only to the person or thing the passage attributes "
    "it to. If the passages do not contain the answer, say exactly: "
    "'The document does not contain this information.' "
    "On the very last line write 'SOURCES: ' followed by the numbers of ALL passages you used, "
    "comma-separated (e.g. 'SOURCES: 1,3')."
)


def answer_language(query: str) -> str:
    """Decided in code, not by the model: Devanagari-dominant question -> Hindi, else English."""
    return "Hindi" if devanagari_ratio(query) >= 0.3 else "English"


def build_prompt(query: str, hits: list[Hit]) -> str:
    ctx = "\n\n".join(f"[{i}] (page {h.page}, section: {h.section})\n{h.text}"
                      for i, h in enumerate(hits, 1))
    if answer_language(query) == "English":
        lang = ("Answer language: ENGLISH. Write the whole answer in English; write Hindi names "
                "in English letters (transliterate carefully) and keep numbers and dates exact.")
    else:
        lang = ("Answer language: HINDI (Devanagari). Keep names, titles and numbers exactly "
                "as written in the passages.")
    return f"Context passages:\n{ctx}\n\nQuestion: {query}\n\n{lang}"


def _call_llm(provider: str, prompt: str) -> str:
    if provider == "anthropic":
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=config.ANTHROPIC_MODEL, max_tokens=400, temperature=0,
            system=SYSTEM, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    if provider == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=config.OPENAI_MODEL, temperature=0,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content
    if provider == "ollama":
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", timeout=config.OLLAMA_TIMEOUT, json={
            "model": config.OLLAMA_MODEL, "stream": False, "keep_alive": "30m",
            # num_ctx: room for 5 Hindi passages (otherwise Ollama may silently truncate the prompt)
            # num_predict: answers are 1-3 sentences, so cap generation to save time
            "options": {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX, "num_predict": 300},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
        if r.status_code != 200:   # show Ollama's own message (e.g. "model ... not found")
            raise RuntimeError(f"Ollama error {r.status_code} for model "
                               f"'{config.OLLAMA_MODEL}': {r.text}")
        return r.json()["message"]["content"]
    raise ValueError(f"Unknown LLM provider: {provider}")


_MARKER = re.compile(r"[\s*_`>#-]*\bsources?\b[\s*_`]*[:：=][\s*_`]*", re.I)
_TAIL_OK = re.compile(r"^[\s\d,;.\[\]()*_`and\u0914\u0930-]*$", re.I)   # digits & separators only


def parse_sources(raw: str, n_hits: int) -> tuple[str, list[int]]:
    """Split 'answer ... SOURCES: 1,3' into (clean answer, [passage numbers]).
    Tolerant of bold markers, full-width colons and 'Sources:' casing; the marker is cut
    from the answer only when what follows is purely numbers, so real text is never lost."""
    last = None
    for last in _MARKER.finditer(raw):
        pass
    if last and _TAIL_OK.match(raw[last.end():]):
        nums = [int(x) for x in re.findall(r"\d+", raw[last.end():]) if 0 < int(x) <= n_hits]
        return raw[:last.start()].strip(), nums
    return raw.strip(), []


def generate(query: str, hits: list[Hit], provider: str | None = None) -> tuple[str, list[Hit]]:
    """Return (answer, hits actually cited). Citations come from stored metadata, never from
    the LLM's own text: the LLM only says *which* passage numbers it used."""
    provider = provider or config.llm_provider()
    if provider == "none":
        return ("[No LLM configured - showing the best-matching passage]\n" + hits[0].text, hits[:1])

    raw = _call_llm(provider, build_prompt(query, hits)).strip()
    answer, nums = parse_sources(raw, len(hits))
    cited = [hits[i - 1] for i in dict.fromkeys(nums)]           # de-duplicated, order kept
    return answer, (cited or hits[:1])                            # fallback: top-1 chunk