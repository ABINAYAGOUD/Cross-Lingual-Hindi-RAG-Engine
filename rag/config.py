"""Central settings. Everything can be overridden with environment variables."""
import os

PDF_PATH = os.getenv("PDF_PATH", "data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")
DB_DIR = os.getenv("DB_DIR", "chroma_db")
COLLECTION = os.getenv("COLLECTION", "kalam_hindi")

# Multilingual embedding model. Alternatives: intfloat/multilingual-e5-small (faster),
# BAAI/bge-m3 (stronger, heavier), intfloat/multilingual-e5-large.
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

# Chunking (characters, sentence-aware). See README for the reasoning.
MAX_CHARS = int(os.getenv("MAX_CHARS", "600"))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", "120"))

TOP_K = int(os.getenv("TOP_K", "5"))


def llm_provider() -> str:
    """anthropic | openai | ollama | none (extractive fallback, no LLM)."""
    p = os.getenv("LLM_PROVIDER", "").lower()
    if p:
        return p
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "none"


ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1200"))   # seconds; CPU-only machines can be slow
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))   # Hindi is token-hungry; avoid silent truncation
