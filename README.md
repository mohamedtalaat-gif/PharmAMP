# PharmAMP

[![Tests](https://github.com/mohamedtalaat-gif/PharmAMP/actions/workflows/tests.yml/badge.svg)](https://github.com/mohamedtalaat-gif/PharmAMP/actions/workflows/tests.yml)

Pharmaceutical-developability metrics for AI-designed antimicrobial peptides.

## Problem

Generative models for antimicrobial peptides (AMPs) — HydrAMP, OmegAMP, and
similar diffusion/VAE-based designers — are optimized and evaluated on potency,
novelty, and diversity. None of the standard evaluation suites, including
[`seqme`](https://github.com/szczurek-lab/seqme), score the property that
actually decides whether a candidate survives past the hit-finding stage:
developability. A peptide with excellent predicted activity that aggregates in
solution or has poor aqueous solubility is not a drug candidate, it's a dead
end (Roberts, Warwicker & Curtis, in Ouyang & Smith, *Computational
Pharmaceutics*, 2015, ch. 7). The scoring approach itself — physicochemical
descriptors combined into a QSPR-style score — follows the general
methodology in Leach, *Molecular Modelling: Principles and Applications*
(2001, ch. 12).

PharmAMP adds that missing layer: sequence-based aggregation-propensity and
solubility scoring, built as `seqme`-compatible metrics, plus a benchmark
harness that runs AMP generators through both the existing potency/diversity
metrics and these new developability ones side by side.

![Pipeline diagram: generative AMP design filters candidates on predicted potency alone and only discovers aggregation or solubility failures after costly wet-lab synthesis; PharmAMP inserts a developability screen before that step.](docs/figures/potency_vs_developability.svg)

## What's here

- `pharmamp/metrics/` — the framework. Two `seqme.core.base.Metric`
  implementations (`AggregationPropensityMetric`, `SolubilityMetric`), built
  on published descriptor logic (Kyte-Doolittle hydrophobicity, net charge,
  hydrophobic moment) rather than a fitted model, for now — see the
  validation results below.
- `pharmamp/benchmark/` — wraps external generators (OmegAMP, plus two
  non-lab null-model baselines — random amino acids and shuffled natural
  AMPs) so their output can be scored with `seqme`'s built-in metrics and
  PharmAMP's developability metrics in one run.
- `app.py` — a Streamlit tool for scoring a batch of candidate sequences
  (pasted or FASTA) and flagging aggregation/solubility risk.

## Install

Works the same way on macOS, Linux, and Windows — only the venv-activation
command differs:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows (cmd/PowerShell): .venv\Scripts\activate
pip install -e .
```

For the benchmark harness, which runs generation through a vendored copy of
OmegAMP, install the extra (same commands on all three platforms — `git` and
`gdown` don't differ by OS):

```
git clone https://github.com/szczurek-lab/OmegAMP.git third_party/OmegAMP
pip install -e ".[benchmark]"
gdown "https://drive.google.com/uc?id=1KO-6Aa7K5_G03DTiwfCa-gPAIOsuxigd" -O models/generative_model.ckpt
gdown "https://drive.google.com/uc?id=1divlvNxsmjYacqb7wb6nK06b8XF8JlOw" -O data/generative-model-data/generative-model-embeddings.h5
```

The checkpoint and embeddings file are OmegAMP's own pretrained weights,
linked from their README — not retrained here. Loading a `.ckpt` means
unpickling it, which can execute arbitrary code if the file isn't what it
claims to be — only fetch it via the exact `gdown` command above. Don't
point `OmegAMPGenerator`'s `checkpoint_path` at a user-supplied file or URL;
that class currently has no integrity check (hash/signature) beyond "the
path exists," which is fine for local use but would need hardening before
any future feature lets someone else choose the checkpoint.

To reproduce the wet-lab validation below, install `pip install -e ".[validate]"`
and place `media-2.xlsx` (from the bioRxiv supplementary materials for
doi:10.64898/2026.09.01.747572) at `data/wetlab-supplement/media-2.xlsx`,
then run `python -m pharmamp.benchmark.validate_against_wetlab`.

`pharmamp/benchmark/generators.py` skips loading the checkpoint's internal
XGBoost activity classifier: `xgboost`'s native `load_model` segfaults when
called in a process that already has `torch`/`pytorch_lightning` loaded.
Isolated this directly — the same call loads fine in a bare Python process
with only `xgboost` imported, and only crashes once `torch` is imported
first, on this machine's `xgboost`/`torch` build combination, independent
of OmegAMP's own code or the model file. That classifier only feeds
`val/amp-probability`, a training-time checkpoint-selection metric never
touched by a bare `.sample()` call, so skipping its load doesn't affect
generation. The fix is applied unconditionally, on every platform — it's a
known macOS/Apple Silicon-specific collision between the two libraries'
bundled OpenMP runtimes, but skipping an unused classifier load costs
nothing on platforms where the collision doesn't happen.

All of `pharmamp/` uses `pathlib` for every path and has no OS-specific
calls, so it's written to run identically on macOS, Linux, and Windows.
Development has been on macOS; CI (the badge above) now runs the test suite
on all three platforms on every push, so cross-platform correctness is
checked automatically rather than just asserted.

## Run the scoring tool

```
streamlit run app.py
```

Or, after the one-time setup above, double-click `run_pharmamp.command` (macOS)
or `run_pharmamp.bat` (Windows) instead of using the terminal. On macOS,
Gatekeeper will block the first launch of a downloaded script — right-click
`run_pharmamp.command` and choose *Open* once to approve it.

## Validation against real wet-lab data

`pharmamp/benchmark/validate_against_wetlab.py` checks the two metrics
against measured safety data for 217 OmegAMP-generated peptides, from
Szymczak, Der Torossian Torres, Soares et al., "Designing antimicrobials
with programmable mechanism and safety" (bioRxiv, 2026,
doi:10.64898/2026.09.01.747572). The supplementary file (`media-2.xlsx`,
CC-BY-ND — not redistributed here) isn't calibration data the metrics were
fit to; it's an independent check of whether the existing heuristics track
real measurements at all.

Result (Spearman, n=215): both metrics correlate significantly with
measured cytotoxicity (CC50) in the expected direction — AggregationPropensity
rho=-0.20 (p=0.004), Solubility rho=+0.19 (p=0.006), only 9.3% of CC50 values
right-censored at the assay ceiling — but not significantly with hemolysis
(HC50) specifically (p=0.09-0.18), where 50.2% of values are censored
(recorded as ">128", meaning no hemolysis at the highest tested
concentration), so that null result carries real uncertainty rather than
being a clean negative. `correlations()` reports the censored fraction
per pair rather than hiding it. More striking: AggregationPropensity
correlates *negatively* with measured β-sheet content (rho=-0.40, p<1e-8) —
the opposite of what its own citation (Chiti & Dobson, 2006, on amyloid-type
aggregation) would predict. The likely explanation: that citation describes
classic amyloid formers, but AMPs are predominantly α-helical/amphipathic,
not β-sheet — so the metric's hydrophobic-patch proxy is apparently tracking
membrane-disruptive/cytotoxic character (where it does correlate, with CC50)
rather than β-sheet-driven amyloid aggregation specifically (where it
doesn't, and runs backwards) — worth a rename or a rework of the metric
once addressed.

## License

MIT.
