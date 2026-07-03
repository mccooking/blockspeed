"""NB-0: the free-parallel-tokens measurement (the roofline check).

Claim under test -- the foundation of the whole project: single-stream LLM
decode is memory-bandwidth-bound, so one forward pass over W new tokens costs
roughly the same wall-clock as W=1 up to a surprisingly large W. If true,
block-parallel generation gets W tokens for the price of one weight-stream.

Method: build a KV cache for a ctx-token prefix (untimed), then time a single
forward pass of W new tokens against that cache. The cache is rebuilt fresh
for every rep so in-place cache mutation across transformers versions cannot
skew results. Attention over the W new tokens is causal here (stock AR
model); a block-diffusion model attends bidirectionally within the block, but
bytes moved and FLOPs are near-identical -- the roofline doesn't care.

Run on the target GPU. A Colab T4 gives the *shape* of the curve (the claim);
H100/B200 later give the record arithmetic. CPU runs are for smoke-testing
the code path only (use --quick).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_CTX_LENS = (512, 2048)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _device_name(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    import platform

    return f"CPU ({platform.processor() or platform.machine()})"


def load_model(model_name: str, device: str | None = None, dtype: torch.dtype | None = None):
    from transformers import AutoModelForCausalLM

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        if device == "cpu":
            dtype = torch.float32
        else:
            # bf16 only on Ampere+ (cc >= 8.0). Turing (T4) runs bf16 via slow
            # emulation, and torch.cuda.is_bf16_supported() still says yes.
            major, _ = torch.cuda.get_device_capability(0)
            dtype = torch.bfloat16 if major >= 8 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return model, device, dtype


@torch.no_grad()
def measure(
    model_name: str = DEFAULT_MODEL,
    widths: tuple[int, ...] = DEFAULT_WIDTHS,
    ctx_lens: tuple[int, ...] = DEFAULT_CTX_LENS,
    n_reps: int = 20,
    n_warmup: int = 5,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Time W-wide decode passes against a cached prefix.

    Returns a JSON-serializable dict; feed it to plot() / save_result().
    """
    import transformers

    model, device, dtype = load_model(model_name, device, dtype)
    vocab = int(model.config.vocab_size)
    rng = torch.Generator().manual_seed(seed)

    rows: list[dict] = []
    for ctx in ctx_lens:
        prefix = torch.randint(0, vocab, (1, ctx), generator=rng).to(device)
        for width in widths:
            new_tokens = torch.randint(0, vocab, (1, width), generator=rng).to(device)
            mask = torch.ones(1, ctx + width, dtype=torch.long, device=device)
            times_s: list[float] = []
            for rep in range(n_warmup + n_reps):
                past = model(prefix, use_cache=True).past_key_values
                _sync(device)
                t0 = time.perf_counter()
                model(new_tokens, past_key_values=past, attention_mask=mask, use_cache=True)
                _sync(device)
                if rep >= n_warmup:
                    times_s.append(time.perf_counter() - t0)
                del past

            med = statistics.median(times_s)
            if len(times_s) >= 4:
                q1, _, q3 = statistics.quantiles(times_s, n=4)
            else:
                q1 = q3 = med
            row = {
                "ctx": ctx,
                "width": width,
                "ms_per_pass": med * 1e3,
                "ms_q1": q1 * 1e3,
                "ms_q3": q3 * 1e3,
                "implied_tok_per_s": width / med,
            }
            rows.append(row)
            if verbose:
                print(
                    f"ctx={ctx:5d}  W={width:4d}   {row['ms_per_pass']:9.2f} ms/pass"
                    f"   -> {row['implied_tok_per_s']:9.0f} tok/s if all W tokens useful"
                )

    for ctx in ctx_lens:
        base = next(r for r in rows if r["ctx"] == ctx and r["width"] == min(widths))
        for r in rows:
            if r["ctx"] == ctx:
                r["cost_ratio_vs_w1"] = r["ms_per_pass"] / base["ms_per_pass"]

    return {
        "experiment": "roofline",
        "model": model_name,
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": _device_name(device),
        "dtype": str(dtype),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "n_reps": n_reps,
        "n_warmup": n_warmup,
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }


def free_widths(result: dict, slack: float = 1.3, baseline_width: int | None = None) -> dict[int, int]:
    """Per ctx: the largest W whose pass costs <= slack x the baseline-width pass.

    baseline_width defaults to the smallest measured width. W=1 uses a
    different kernel family (matrix-vector) than W>=2 (matrix-matrix), so
    comparing against W=2 as well guards the verdict from that discontinuity.
    """
    out: dict[int, int] = {}
    for ctx in sorted({r["ctx"] for r in result["rows"]}):
        rows = [r for r in result["rows"] if r["ctx"] == ctx]
        bw = baseline_width if baseline_width is not None else min(r["width"] for r in rows)
        base = next((r for r in rows if r["width"] == bw), None)
        if base is None:
            continue
        for r in rows:
            if r["ms_per_pass"] <= slack * base["ms_per_pass"]:
                out[ctx] = max(out.get(ctx, 0), r["width"])
    return out


def verdict(result: dict) -> str:
    widths = sorted({r["width"] for r in result["rows"]})
    lines = [f"{result['model']} on {result['device']} ({result['dtype']}):"]
    baselines = widths[:1] if len(widths) < 2 else widths[:2]
    for bw in baselines:
        for slack in (1.3, 2.0):
            for ctx, w in sorted(free_widths(result, slack, baseline_width=bw).items()):
                lines.append(
                    f"  ctx {ctx}: up to {w} tokens per pass at <= {slack:.1f}x the cost of a W={bw} pass"
                )
    return "\n".join(lines)


def plot(result: dict, out_png: str | Path | None = None):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ctxs = sorted({r["ctx"] for r in result["rows"]})
    for ctx in ctxs:
        rows = sorted((r for r in result["rows"] if r["ctx"] == ctx), key=lambda r: r["width"])
        w = [r["width"] for r in rows]
        ax1.plot(w, [r["ms_per_pass"] for r in rows], marker="o", label=f"ctx {ctx}")
        ax1.fill_between(w, [r["ms_q1"] for r in rows], [r["ms_q3"] for r in rows], alpha=0.2)
        ax2.plot(w, [r["cost_ratio_vs_w1"] for r in rows], marker="o", label=f"ctx {ctx}")

    widths = sorted({r["width"] for r in result["rows"]})
    ax2.plot(widths, [x / widths[0] for x in widths], "k:", label="compute-bound (cost = W)")
    ax2.axhline(1.0, color="gray", ls="--", lw=1, label="perfectly free")

    for ax, ylabel in ((ax1, "ms per forward pass"), (ax2, "cost ratio vs W=1")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels([str(x) for x in widths])
        ax.set_xlabel("tokens per pass (W)")
        ax.set_ylabel(ylabel)
        ax.legend()
    ax2.set_yscale("log", base=2)
    fig.suptitle(
        f"Decode-pass cost vs width - {result['model']} - {result['device']} ({result['dtype']})"
    )
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    return fig


def save_result(result: dict, results_dir: str | Path = "results", make_plot: bool = True) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", result["device"]).strip("-").lower()
    stamp = result["timestamp_utc"].replace(":", "").replace("-", "")[:13]
    base = results_dir / f"roofline_{slug}_{stamp}"
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(result, f, indent=2)
    if make_plot:
        plot(result, out_png=base.with_suffix(".png"))
    return base.with_suffix(".json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    p.add_argument("--ctx", type=int, nargs="+", default=list(DEFAULT_CTX_LENS))
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", default="results")
    p.add_argument(
        "--quick",
        action="store_true",
        help="tiny CPU-friendly smoke test of the code path (not a real measurement)",
    )
    args = p.parse_args()

    if args.quick:
        args.model = "sshleifer/tiny-gpt2"
        args.widths = [1, 4, 16]
        args.ctx = [64, 128]
        args.reps = 3
        args.warmup = 1

    result = measure(
        model_name=args.model,
        widths=tuple(args.widths),
        ctx_lens=tuple(args.ctx),
        n_reps=args.reps,
        n_warmup=args.warmup,
    )
    print()
    print(verdict(result))
    path = save_result(result, args.out)
    print(f"\nSaved: {path} (+ .png)")


if __name__ == "__main__":
    main()
