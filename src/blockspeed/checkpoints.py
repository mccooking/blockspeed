"""Pinned Hugging Face checkpoints for the validation ladder.

Single source of truth: notebooks refer to these keys, never to raw repo ids.
Run  python -m blockspeed.checkpoints  after cloning (and after every edit)
so NB-2/NB-3 never stall on a dead link.
"""

from __future__ import annotations

CHECKPOINTS: dict[str, str] = {
    # roofline subjects / AR parents and baselines
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "tiny-gpt2": "sshleifer/tiny-gpt2",  # CPU smoke tests only
    # judges
    "gpt2": "gpt2",
    "gpt2-large": "gpt2-large",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
    # public diffusion teachers (NB-2 / NB-3)
    "mdlm-owt": "kuleshov-group/mdlm-owt",
    "bd3lm-block4": "kuleshov-group/bd3lm-owt-block_size4",
    "bd3lm-block8": "kuleshov-group/bd3lm-owt-block_size8",
    "bd3lm-block16": "kuleshov-group/bd3lm-owt-block_size16",
    # few-step students / duality line (NB-2)
    "duo-base": "s-sahoo/duo",
    "duo-distilled": "s-sahoo/duo-distilled",
}


def verify(ids: dict[str, str] | None = None, token: str | None = None) -> dict[str, str]:
    """Check every pinned id resolves on the Hub. Returns {key: 'ok' | error}."""
    from huggingface_hub import model_info

    ids = ids if ids is not None else CHECKPOINTS
    status: dict[str, str] = {}
    for key, repo_id in ids.items():
        try:
            info = model_info(repo_id, token=token)
            status[key] = f"ok ({info.id})"
        except Exception as e:  # gated, missing, network -- record and move on
            status[key] = f"ERROR: {type(e).__name__}: {e}"
    return status


def main() -> None:
    status = verify()
    width = max(len(k) for k in status)
    bad = 0
    for key, s in status.items():
        print(f"  {key:<{width}}  {s}")
        bad += s.startswith("ERROR")
    print(f"\n{len(status) - bad}/{len(status)} checkpoints resolve")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
