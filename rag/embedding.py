"""Multilingual embedding wrapper (handles the E5 'query:'/'passage:' prefixes)."""
from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # E5 models were trained with these prefixes; omitting them hurts retrieval.
        self.is_e5 = "e5" in model_name.lower()

    def _enc(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 50
        )

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._enc([f"passage: {t}" for t in texts] if self.is_e5 else texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._enc([f"query: {text}" if self.is_e5 else text])[0]

    def token_lengths(self, texts: list[str]) -> list[int]:
        tok = self.model.tokenizer
        return [len(tok(t)["input_ids"]) for t in texts]
