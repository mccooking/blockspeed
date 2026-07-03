"""The honesty harness: every quality number in this project flows through here.

Rule (research.md R6): judge-PPL is gameable by low-entropy degenerate text,
so it is never reported without sample entropy next to it. Run self_test()
once per environment before trusting any measurement.
"""

from .entropy import token_entropy
from .judge import Judge
from .plots import quality_vs_passes


def self_test(**kwargs) -> bool:
    from .selftest import run

    return run(**kwargs)


__all__ = ["Judge", "token_entropy", "quality_vs_passes", "self_test"]
