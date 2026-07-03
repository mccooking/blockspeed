"""Sample entropy: the anti-gaming guard for judge-PPL.

Empirical unigram entropy (bits/token) of generated samples. Healthy natural
English under the GPT-2 BPE pools around ~7.3-8.1 bits (FS-DFM's reference
range); degenerate repetition collapses toward 0. A model that "wins" on
judge-PPL while its entropy craters is gaming the judge, not writing.
"""

from __future__ import annotations

import math
from collections import Counter


def _entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def token_entropy(
    texts: list[str],
    tokenizer_name: str = "gpt2",
    tokenizer=None,
) -> dict:
    """Unigram token entropy, pooled across samples and averaged per sample."""
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    pooled: Counter = Counter()
    per_sample: list[float] = []
    for text in texts:
        tokens = tokenizer.encode(text)
        pooled.update(tokens)
        per_sample.append(_entropy_bits(Counter(tokens)))

    return {
        "tokenizer": tokenizer_name,
        "pooled_entropy_bits": _entropy_bits(pooled),
        "mean_sample_entropy_bits": sum(per_sample) / len(per_sample) if per_sample else 0.0,
        "per_sample_entropy_bits": per_sample,
        "n_tokens": sum(pooled.values()),
    }
