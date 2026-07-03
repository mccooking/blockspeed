# blockspeed

Can a language model generate text in parallel blocks — many tokens per forward pass instead of one — without losing quality? This repo is the validation ladder for that bet, sized to run on free Colab GPUs. Each rung has a pass/fail criterion fixed **before** the experiment runs.

| # | Notebook | Question | Green means |
|---|---|---|---|
| 0 | `00_roofline` | Is a W-token-wide decode pass really ~the cost of W=1 on real silicon? | Pass cost near-flat to W≈32–64 (memory-bound decode confirmed) |
| 1 | `01_harness` | Can we measure quality honestly? | Self-test passes: judge-PPL ranks real ≫ shuffled; entropy catches degenerate text that games the judge |
| 2 | `02_fewstep_repro` | Do published few-step diffusion results hold on *our* harness? | Distilled student at 8–16 steps ≈ its many-step teacher on judge-PPL, at healthy entropy |
| 3 | `03_distill_ladder` | Does block structure × few-step distillation compose? | ≥2–4 tokens/pass at ~teacher quality on a ~110M public teacher; the 8→4→2→1-step quality curve |

Standing rules: no quality number without its steps-per-block **and** entropy; no speed number without its quality; every environment runs the harness self-test before reporting anything.

## Layout

```
src/blockspeed/
  roofline.py      # NB-0: pass-cost vs width measurement
  harness/         # judge-PPL + entropy + quality-vs-passes plots + self-test
  checkpoints.py   # pinned HF repo ids (run `python -m blockspeed.checkpoints` to verify)
notebooks/         # thin Colab drivers; logic stays in the package
results/           # committed JSON + PNG, one pair per measurement
```

## Quickstart

```bash
pip install -e .
python -m blockspeed.harness.selftest        # calibrate instruments (CPU ok)
python -m blockspeed.checkpoints             # verify pinned checkpoints resolve
python -m blockspeed.roofline --quick        # smoke-test the code path (CPU)
python -m blockspeed.roofline                # real measurement (needs a GPU)
```

On Colab: open a notebook from `notebooks/`, run top to bottom, commit the `results/` pair it produces.
