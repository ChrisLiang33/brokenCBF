# Class paper draft

10-page sole-author write-up for the Privileged Robotics class.
Based on the proposal (March 24, 2026) and the actual delivered work
documented in `LOG.md`.

## Files

- `main.tex` — paper source.
- `references.bib` — bibliography (9 entries, all real citations).
- `figures/` — figure PDFs. `placeholder_baseline_sweep.pdf` is a stub.

## Compile

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

(Or `latexmk -pdf main.tex` if you have it.)

## Method-name placeholder

Currently `\methodname{}` is defined as `AR-CBF` (Adaptive Robustness CBF)
in `main.tex` at the top of the preamble. To rename, change one line:
```latex
\newcommand{\methodname}{\textsc{YOUR-NAME}}
```

## Outstanding TODOs (blocked on the lab machine)

1. **Regenerate Figure 1** from `logs/baseline_eval/baseline.csv`. The
   placeholder lives at `figures/placeholder_baseline_sweep.pdf`. Suggested
   plot: scatter of fall rate vs. episode-mean φ for all 24 hand-tuned
   configs + the trained teacher; same data the original `eval_baseline.py`
   already produces in `baseline.png`.
2. **Add a figure for the 5-axis OOD result** (Section 4.3). A bar chart of
   combined-failure-rate margin per axis (Slippery, HighDist, HeavyCOM,
   FastObs, DensePack, Compound) reads cleaner than the current Table 2
   alone.
3. **Add a hero figure** for §1 — block diagram of the architecture (priv
   obs → CNN encoder → π → robust CBF QP → off-the-shelf locomotion → robot)
   with one frame from a simulation rollout overlaid.
4. **(Optional) training-curve figure** for §3.5 — base_contact rate over
   PPO iterations across v2.6 and the locked-planner ablation; visualises
   the regression from §4.4 quantitatively.
5. **Method name.** Replace `AR-CBF` with whatever you and Cosner agree on.

## Outstanding TODOs (not blocked)

- **Numbers in the abstract** are conservative copies of the in-distribution
  result. If you want the abstract to mention OOD axes specifically (e.g.
  "+6.9pp in-distribution and +5–11pp on four privileged-observation OOD
  axes"), that line is in `main.tex` and trivially editable.
- **Discount the +6.9pp claim** if the locked-planner ablation lands as
  expected and the planner-mid-switch caveat needs to be promoted to the
  headline rather than just §4.4.

## What was deliberately left out

- Hardware experiments — no hardware data exists yet; framed as future
  work in §5.
- Student distillation — gated on the teacher result; flagged in §2.3 and
  §6 as the obvious next step.
- v2.7 / v2.8 / v2.9 detailed timeline — only the v2.8 locked-planner result
  appears in §4.4 because that is the one ablation with a clean
  one-knob diagnostic. The v2.7 wider-DR and v2.9 reward-shaping work is
  mentioned under §5 limitations rather than as separate ablations.
