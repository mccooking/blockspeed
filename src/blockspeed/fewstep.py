"""NB-2: few-step reproduction on public Duo checkpoints (uniform-state diffusion).

Ports the *authors'* ancestral sampler for DUO (github.com/s-sahoo/duo:
trainer_base.py generate_samples/_ancestral_update + algo.py
DUO_BASE._posterior_from_x0) into a standalone loop against their HF
checkpoints, so generation follows the published algorithm exactly while
measurement runs on our harness. Defaults mirror the repo's HF sampling
command: log-linear noise (eps 1e-3), linear time grid down to 1e-5, greedy
noise removal.

One substitution: a pure-PyTorch shim stands in for the flash_attn package.
The DUO diffusion path only uses flash-attn's *torch* rotary helper and
F.scaled_dot_product_attention (the CUDA kernels are only reached by the
causal/AR blocks, which diffusion checkpoints never instantiate), so results
are identical -- and it runs on pre-Ampere GPUs (Colab T4) where flash-attn
cannot install. The checkpoint's remote code also needs `einops` and
`omegaconf` installed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

DUO_TEACHER = "s-sahoo/duo"
DUO_DISTILLED = "s-sahoo/duo-distilled"


# --------------------------------------------------------------------------
# flash-attn shim
# --------------------------------------------------------------------------

def _apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """Pure-torch rotary embedding, matching the non-interleaved semantics of
    flash_attn.layers.rotary.apply_rotary_emb_torch (flash-attn, BSD-3).

    x: (batch, seq, heads, dim); cos/sin: (seq, rotary_dim/2).
    """
    assert not interleaved, "DUO uses the non-interleaved path"
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(-2)  # (seq, 1, ro_dim)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(-2)
    x_ro, x_pass = x[..., :ro_dim], x[..., ro_dim:]
    x1, x2 = x_ro.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    out = torch.cat([x_ro * cos + rotated * sin, x_pass], dim=-1)
    return out.to(x.dtype)


def install_flash_attn_shim() -> bool:
    """Make `import flash_attn` resolve without the CUDA package.

    Returns True if the shim was installed, False if real flash-attn exists.
    The CUDA-only entry points raise if reached -- the DUO diffusion path
    never calls them.
    """
    try:
        import flash_attn  # noqa: F401

        return False
    except Exception:
        pass

    def _cuda_only(*_args, **_kwargs):
        raise NotImplementedError(
            "Real flash-attn kernel requested (causal/AR block). "
            "The DUO diffusion path should never reach this."
        )

    import importlib.machinery

    def _make_module(name: str, is_package: bool = False) -> types.ModuleType:
        # A real ModuleSpec keeps importlib.util.find_spec() happy
        # (transformers probes it); absent dist metadata still makes
        # transformers treat flash-attn as unavailable, which is correct.
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=is_package)
        if is_package:
            mod.__path__ = []
        return mod

    fa = _make_module("flash_attn", is_package=True)
    fa_layers = _make_module("flash_attn.layers", is_package=True)
    fa_rotary = _make_module("flash_attn.layers.rotary")
    fa_iface = _make_module("flash_attn.flash_attn_interface")
    fa_rotary.apply_rotary_emb_torch = _apply_rotary_emb_torch
    fa_rotary.apply_rotary_emb_qkv_ = _cuda_only
    fa_iface.flash_attn_varlen_qkvpacked_func = _cuda_only
    fa.layers = fa_layers
    fa_layers.rotary = fa_rotary
    fa.flash_attn_interface = fa_iface
    fa.__version__ = "0.0.0-blockspeed-shim"
    sys.modules["flash_attn"] = fa
    sys.modules["flash_attn.layers"] = fa_layers
    sys.modules["flash_attn.layers.rotary"] = fa_rotary
    sys.modules["flash_attn.flash_attn_interface"] = fa_iface
    return True


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------

def load_duo(repo_id: str = DUO_DISTILLED, device: str | None = None):
    """Load a DUO HF checkpoint (fp32, faithful to the authors' eval)."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    install_flash_attn_shim()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    try:
        model = AutoModelForMaskedLM.from_pretrained(repo_id, trust_remote_code=True)
    except AttributeError as e:
        if "all_tied_weights_keys" not in str(e):
            raise
        # The DUO remote code predates transformers 5.x, whose loader expects
        # this attribute. DUO ties no weights, so an empty dict is correct.
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        cls = get_class_from_dynamic_module(
            cfg.auto_map["AutoModelForMaskedLM"], repo_id
        )
        cls.all_tied_weights_keys = property(lambda self: {})
        model = cls.from_pretrained(repo_id)
    model.to(device)
    model.eval()
    return model, tokenizer, device


# --------------------------------------------------------------------------
# the authors' sampler, standalone
# --------------------------------------------------------------------------

def _model_logits(model, x, sigma):
    # Remote code returns MaskedLMOutput on transformers 4.x but a raw tensor
    # on 5.x (use_return_dict deprecation changes its return_dict default).
    out = model(input_ids=x, timesteps=sigma)
    if torch.is_tensor(out):
        return out
    if hasattr(out, "logits"):
        return out.logits
    return out[0]


def _sample_categorical(probs):
    # trainer_base.sample_categorical: Gumbel-max via exponential race
    gumbel_norm = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / gumbel_norm).argmax(dim=-1)


def _uniform_posterior(p_x0, xt, alpha_s, alpha_t, vocab_size):
    # algo.py DUO_BASE._posterior_from_x0 (uniform state). alphas: (B, 1, 1).
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, vocab_size).to(p_x0.dtype)
    numerator = (
        alpha_t * vocab_size * p_x0 * xt_one_hot
        + (alpha_ts - alpha_t) * xt_one_hot
        + d_alpha * p_x0
        + (1 - alpha_ts) * (1 - alpha_s) / vocab_size
    )
    denominator = alpha_t * vocab_size * torch.gather(
        p_x0, -1, xt[..., None]
    ) + (1 - alpha_t)
    return numerator / denominator


@torch.no_grad()
def sample_duo(
    model,
    num_steps: int,
    num_samples: int = 4,
    seq_len: int = 1024,
    noise_eps: float = 1e-3,
    sampling_eps: float = 1e-5,
    greedy_tail: bool = True,
    use_float64: bool = False,
    seed: int | None = None,
    device: str | None = None,
):
    """Ancestral sampling for DUO uniform-state checkpoints.

    Faithful port of generate_samples (predictor='ancestral', linear grid,
    log-linear noise, noise_removal='greedy'). use_float64 matches the repo's
    high-precision posterior option but costs ~2x memory; fp32 is the
    default here and is recorded in results.
    """
    device = device or next(model.parameters()).device
    vocab_size = model.config.vocab_size
    if seed is not None:
        torch.manual_seed(seed)

    def alpha(t):  # LogLinear schedule: alpha_t = 1 - (1 - eps) * t
        return 1 - (1 - noise_eps) * t

    x = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    timesteps = torch.linspace(1.0, sampling_eps, num_steps + 1, device=device)

    for i in range(num_steps):
        t = timesteps[i].expand(num_samples, 1)
        s = timesteps[i + 1].expand(num_samples, 1)
        alpha_t, alpha_s = alpha(t), alpha(s)
        sigma = -alpha_t.log().mean(-1)  # _process_sigma: (B,)
        logits = _model_logits(model, x, sigma)
        p_x0 = logits.log_softmax(-1).exp()
        if use_float64:
            p_x0 = p_x0.to(torch.float64)
        q_xs = _uniform_posterior(
            p_x0, x, alpha_s[..., None], alpha_t[..., None], vocab_size
        )
        x = _sample_categorical(q_xs)

    if greedy_tail:  # sampling.noise_removal='greedy'
        t0 = timesteps[-1].expand(num_samples, 1)
        sigma0 = -alpha(t0).log().mean(-1)
        x = _model_logits(model, x, sigma0).argmax(-1)
    return x


# --------------------------------------------------------------------------
# experiment orchestration
# --------------------------------------------------------------------------

def run_config(
    model,
    tokenizer,
    label: str,
    num_steps: int,
    num_samples: int = 16,
    batch_size: int = 4,
    seq_len: int = 1024,
    seed: int | None = 0,
    verbose: bool = True,
    **sample_kwargs,
) -> dict:
    """Generate num_samples texts at num_steps; returns texts + timing."""
    texts: list[str] = []
    t_start = time.perf_counter()
    batch_idx = 0
    while len(texts) < num_samples:
        n = min(batch_size, num_samples - len(texts))
        tokens = sample_duo(
            model,
            num_steps,
            num_samples=n,
            seq_len=seq_len,
            seed=None if seed is None else seed + batch_idx,
            **sample_kwargs,
        )
        texts.extend(tokenizer.batch_decode(tokens))
        batch_idx += 1
    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"{label} @ {num_steps} steps: {num_samples} samples in {elapsed:.0f}s")
    return {
        "label": label,
        "num_steps": num_steps,
        "seq_len": seq_len,
        "num_samples": num_samples,
        "passes_per_token": num_steps / seq_len,
        "gen_seconds": elapsed,
        "texts": texts,
    }


def evaluate_configs(
    configs: list[dict],
    judge=None,
    judge_model: str = "gpt2-large",
    entropy_tokenizer: str = "gpt2",
    verbose: bool = True,
) -> list[dict]:
    """Judge-PPL + entropy for each generated config (harness rule: paired)."""
    from .harness import Judge, token_entropy

    if judge is None:
        judge = Judge(judge_model)
    rows = []
    for cfg in configs:
        ppl = judge.ppl(cfg["texts"])
        ent = token_entropy(cfg["texts"], entropy_tokenizer)
        row = {k: v for k, v in cfg.items() if k != "texts"}
        row["judge"] = judge.model_name
        row["judge_ppl"] = ppl["mean_ppl"]
        row["pooled_entropy_bits"] = ent["pooled_entropy_bits"]
        rows.append(row)
        if verbose:
            print(
                f"{cfg['label']} @ {cfg['num_steps']:5d} steps: "
                f"judge-PPL {row['judge_ppl']:8.1f}  "
                f"entropy {row['pooled_entropy_bits']:5.2f} bits"
            )
    return rows


def plot(rows: list[dict], out_png: str | Path | None = None):
    from .harness import quality_vs_passes

    series = []
    for label in sorted({r["label"] for r in rows}):
        pts = sorted(
            (r for r in rows if r["label"] == label),
            key=lambda r: r["passes_per_token"],
        )
        series.append(
            {
                "label": label,
                "passes_per_token": [p["passes_per_token"] for p in pts],
                "quality": [p["judge_ppl"] for p in pts],
                "ls": "-" if len(pts) > 1 else "",
            }
        )
    return quality_vs_passes(
        series,
        ylabel="judge perplexity",
        title="Few-step reproduction: quality vs passes (Duo checkpoints)",
        higher_is_better=False,
        out_png=out_png,
    )


def save_result(
    rows: list[dict],
    configs: list[dict] | None = None,
    results_dir: str | Path = "results",
    device_name: str = "",
) -> Path:
    """Persist eval rows (+ raw sample texts for audit) and the curve plot."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace(":", "")
        .replace("-", "")[:13]
    )
    slug = re.sub(r"[^A-Za-z0-9]+", "-", device_name).strip("-").lower() or "unknown"
    base = results_dir / f"fewstep_{slug}_{stamp}"
    payload = {
        "experiment": "fewstep_repro",
        "device": device_name,
        "torch": torch.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
        "samples": {
            f"{c['label']}@{c['num_steps']}": c["texts"] for c in (configs or [])
        },
    }
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    plot(rows, out_png=base.with_suffix(".png"))
    return base.with_suffix(".json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true",
                   help="tiny CPU smoke test of the full code path")
    args = p.parse_args()
    if not args.quick:
        p.error("only --quick is supported from the CLI; use the notebook for real runs")

    model, tokenizer, device = load_duo(DUO_DISTILLED)
    cfg = run_config(
        model, tokenizer, "duo-distilled", num_steps=2,
        num_samples=2, batch_size=2, seq_len=32,
    )
    print(repr(cfg["texts"][0][:120]))
    rows = evaluate_configs([cfg], judge_model="gpt2")
    print("\nsmoke OK -- code path runs end to end (numbers are meaningless at this size)")


if __name__ == "__main__":
    main()
