"""Judge perplexity: score generated text under a fixed reference LM.

Lower judge-PPL = the judge finds the text more fluent. Alone it is gameable
(degenerate repetition scores deceptively well) -- always pair with
entropy.token_entropy. Judges: gpt2 for CPU self-tests, gpt2-large for
literature comparability, smollm2-1.7b / qwen3-1.7b for stronger judging.
"""

from __future__ import annotations

import math

import torch


class Judge:
    def __init__(
        self,
        model_name: str = "gpt2-large",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        max_length: int = 1024,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float32 if device == "cpu" else torch.float16
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def nll(self, text: str) -> tuple[float, int]:
        """Mean per-token negative log-likelihood, and the token count scored."""
        ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return float("nan"), 0
        loss = self.model(ids, labels=ids).loss
        return float(loss), ids.shape[1] - 1

    @torch.no_grad()
    def ppl(self, texts: list[str]) -> dict:
        """Per-text and pooled perplexity. Report alongside token_entropy, always."""
        per_text: list[float] = []
        total_nll = 0.0
        total_tokens = 0
        for text in texts:
            loss, n = self.nll(text)
            if n == 0:
                continue
            per_text.append(math.exp(loss))
            total_nll += loss * n
            total_tokens += n
        return {
            "judge": self.model_name,
            "mean_ppl": sum(per_text) / len(per_text),
            "pooled_ppl": math.exp(total_nll / total_tokens),
            "per_text_ppl": per_text,
            "n_texts": len(per_text),
            "n_tokens": total_tokens,
        }
