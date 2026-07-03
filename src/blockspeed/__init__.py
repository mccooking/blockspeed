"""blockspeed: validation ladder for block-parallel fast LLM generation.

NB-0  roofline      -- is a W-token-wide decode pass really ~the cost of W=1?
NB-1  harness       -- judge-PPL + sample entropy, calibrated by a self-test
NB-2  few-step repro -- published few-step checkpoints re-measured on our harness
NB-3  distill ladder -- 8 -> 4 -> 2 -> 1 steps/block on a public teacher

Heavy imports (torch/transformers) happen inside submodules, not here.
"""

__version__ = "0.1.1"
