"""Calibrate the instruments before trusting any measurement.

Three known inputs, three expected readings:
  1. real English         -> judge-PPL clearly better than shuffled English
  2. shuffled English     -> judge-PPL clearly worse than real
  3. degenerate repetition -> entropy near zero (while its judge-PPL may look
     deceptively GOOD -- that is the trap this harness exists to catch)

Runs on CPU with the small gpt2 judge in ~a minute. Run once per environment
(local, Colab, rented box) before reporting any number from that environment.

Usage: python -m blockspeed.harness.selftest [--judge gpt2] [--device cpu]
"""

from __future__ import annotations

import argparse
import random

from .entropy import token_entropy
from .judge import Judge

REAL_TEXTS = [
    "The train left the station a few minutes late, but the driver made up "
    "time on the long straight stretch through the valley. Most passengers "
    "were asleep by then, their bags wedged under the seats, and the lights "
    "of the small towns slid past the windows like slow sparks.",
    "To make a decent stock, start with cold water and never let it reach a "
    "rolling boil. Skim the surface as it warms, add the vegetables late, "
    "and be patient: the difference between a thin broth and a rich one is "
    "mostly time, not ingredients.",
    "The committee reviewed the proposal for nearly two hours before voting. "
    "Several members raised concerns about the budget, particularly the cost "
    "of maintaining the new building, but in the end the measure passed with "
    "a comfortable majority and the meeting moved on to routine business.",
    "Glaciers move more like thick honey than like ice cubes. Under enough "
    "pressure, the crystalline structure at the base begins to deform and "
    "slide, so the whole mass creeps downhill a few centimeters a day, "
    "grinding the rock beneath it into fine gray flour.",
    "She kept the shop open through the winter even though hardly anyone "
    "came. Regulars knew to knock on the side door after dark, and she would "
    "sell them bread from the morning batch at half price rather than see it "
    "go stale on the shelf.",
    "The instructions were simple enough on paper: connect the pump, prime "
    "the line, and check the pressure twice before opening the main valve. "
    "In practice the gauge stuck, the fittings leaked, and the whole job "
    "took most of the afternoon and two trips to the hardware store.",
]


def _shuffle_words(text: str, seed: int) -> str:
    words = text.split()
    rng = random.Random(seed)
    rng.shuffle(words)
    return " ".join(words)


def run(
    judge_model: str = "gpt2",
    tokenizer_name: str = "gpt2",
    device: str | None = None,
    verbose: bool = True,
) -> bool:
    real = REAL_TEXTS
    shuffled = [_shuffle_words(t, seed=i) for i, t in enumerate(real)]
    degenerate = ["the " * 150, "very very " * 75, "and then and then " * 40]

    judge = Judge(judge_model, device=device)
    ppl_real = judge.ppl(real)
    ppl_shuffled = judge.ppl(shuffled)
    ppl_degenerate = judge.ppl(degenerate)
    ent_real = token_entropy(real, tokenizer_name)
    ent_degenerate = token_entropy(degenerate, tokenizer_name)

    checks = [
        (
            "judge ranks real text far above shuffled (PPL real < PPL shuffled / 2)",
            ppl_real["mean_ppl"] < ppl_shuffled["mean_ppl"] / 2,
        ),
        (
            "degenerate repetition has near-zero entropy (< 2.5 bits)",
            ent_degenerate["pooled_entropy_bits"] < 2.5,
        ),
        (
            "real text has healthy entropy (> 5 bits)",
            ent_real["pooled_entropy_bits"] > 5.0,
        ),
    ]

    if verbose:
        print(f"judge = {judge_model} on {judge.device}, tokenizer = {tokenizer_name}\n")
        print(f"  mean judge-PPL   real: {ppl_real['mean_ppl']:10.1f}")
        print(f"  mean judge-PPL   shuffled: {ppl_shuffled['mean_ppl']:6.1f}")
        print(f"  mean judge-PPL   degenerate: {ppl_degenerate['mean_ppl']:4.1f}")
        print(f"  pooled entropy   real: {ent_real['pooled_entropy_bits']:6.2f} bits")
        print(f"  pooled entropy   degenerate: {ent_degenerate['pooled_entropy_bits']:.2f} bits\n")
        if ppl_degenerate["mean_ppl"] < ppl_real["mean_ppl"]:
            print(
                "  NOTE: degenerate text out-scores real text on judge-PPL alone.\n"
                "  That is the trap. The entropy column is what catches it.\n"
            )
        for name, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    passed = all(ok for _, ok in checks)
    if verbose:
        print(f"\nself-test: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate the blockspeed honesty harness.")
    p.add_argument("--judge", default="gpt2")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    ok = run(judge_model=args.judge, tokenizer_name=args.tokenizer, device=args.device)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
