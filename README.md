# ConsensusCourt: Procedural Evidence Arbitration for Scientific Claims in Multi-Agent Research Pipelines

**Preprint (v0.4.17, September 21, 2026)** · Wei Wang · Shanghai Hongene Biologics Tech Co., Ltd.

This repository is the complete open-source companion to the preprint:
manuscript sources, the court/baseline implementations under test, every
deterministic experiment pipeline behind the paper's numbers, and the
machine-readable reports they produce.

## Abstract (abridged)

Multi-agent LLM research pipelines generate streams of evidence — retrieved
literature, computational results, agent opinions — that frequently conflict.
ConsensusCourt borrows its procedural guarantees from the courtroom rather
than voting theory:

- **Asymmetric burden of proof (ABP)** — every claim is presumed innocent
  until a detected *contradiction pair* convicts it;
- **Source-pedigree evidence scoring** — source, recency, and relevance, with
  the confidence channel structurally decoupled from conviction;
- **Three-state citation verification** (pass / fail / unknown) — holds the
  clean-claim false-positive rate at 0 under verification failure rates up to
  100%, where the common two-state policy convicts up to 100% of innocent
  claims;
- **Escalation layer** — uncertain or unadjudicable cases route to human
  review instead of guessing.

Headline results: a counting law (log-log slopes 0.942 / −0.857 against
expected 1.0 / −1.0), the network-failure experiment above, and a
conviction–confidence decoupling invariant preserved exactly under every
scorer ablation. On the deterministic 954-case DDB benchmark the full court
reaches recall = 1.0 with FPR = 0.0 while majority-vote, Elo, and single-judge
baselines fail on at least one Pareto side at every scanned threshold.

## Repository layout

```
paper/                  manuscript sources + compiled PDF + data provenance
  main.tex main.bbl references.bib ConsensusCourt.pdf DATA_PROVENANCE.md
code/
  core/                 implementations under test (court, baselines, rubric)
    consensus_court.py consensus_court_omics.py baseline_arbiter.py rubric_engine.py
  scripts/              one pipeline per paper table/figure
    run_v03_regression.sh          <- single entry point for the full suite
    run_ablation_v03.py            (DDB-L1/L1.5/L4 + court + scorer ablations)
    run_baseline_comparison.py     (S6.2 controlled comparison vs 3 baselines)
    run_baseline_sensitivity.py    (S6.2 49-grid Pareto scan, RL1/RL2/RL3)
    run_robustness_experiments.py  (S6.5 Thm 2 + Thm 1-B + streaming + noise)
    run_advantage_decomposition.py (S5.3 Thm 1-A identity + kappa)
    run_role_weight_calibration.py (S6.5 13-role +/-20% flip-rate)
    run_network_failure_experiment.py (S6.5 three-state vs two-state FPR)
    run_case_study.py              (S6.9 per-case court timelines)
    run_llm_arbiter_experiment.py  (S6.10 LLM arbiter ablation)
    run_llm_debate_experiment.py   (S6.10 PROClaim/DebateCV-style debate ablation)
  tests/                pytest suites (418 tests) mirroring the pipelines above
  data/                 machine-readable reports (every number in the paper)
                        + frozen benchmark case sets (l1/l15/l2/l4)
```

## Reproducing every number in the paper

**Full-suite reproduction (as wired in the parent application).** The paper's
regression entry point, `run_v03_regression.sh`, is written for the layout of
the parent backend workspace (`backend/tests/...`): copy `code/tests`,
`code/scripts` and `code/core` into your `backend/tests/` and
`backend/app/core/virtual_lab/` trees and run
`bash backend/tests/scripts/run_v03_regression.sh` from the backend root.
Requirements: Python >= 3.11 (the ORM layer uses `typing.ParamSpec`),
`pytest`, and the parent project's runtime dependencies. Exit codes: `0` all
green; `1` v0.3 suite failure; `2` legacy regression failure; `3` ablation
red-line violation.

**Standalone check (no parent workspace).** The benchmark generators, the
court/rubric/baseline implementations, and the frozen reports are self-
contained enough that the deterministic benchmark itself
(`code/tests/ddb/l1_generator.py`, seed = 42) and every headline number can be
inspected directly from `code/data/*.json`; `paper/DATA_PROVENANCE.md` maps
each manuscript figure/table to its report file.

The suite is deterministic (seed = 42) and zero-LLM for all main results; the
two S6.10 LLM-side ablations require a hosted model key and their frozen
reports ship in `code/data/` regardless.

Pre-registered red lines enforced by the suite (see paper S8):

- FPR non-increase / recall non-decrease across v0.2 -> v0.3 -> v0.4 rubrics;
- negative-basis (L1.5) full_v03/v04 FPR = 0;
- intensity-label (L4) consistency;
- court ablation monotonicity (ABP necessity, reviewer/BOSS signal direction);
- scorer-ablation FPR/recall monotonicity;
- three-state FPR = 0 at every failure-rate grid point;
- per-case advantage-decomposition identity closing exactly (residual = 0).

All outputs regenerate as JSON into `backend/tests/ddb/` (in the parent
workspace) or are comparable against the frozen copies in `code/data/`.

## Provenance and integrity

`paper/DATA_PROVENANCE.md` maps every quantitative claim in the manuscript to
its generating script, seed, and report file. 2026 citations were
metadata-verified against arXiv/PubMed on 2026-09-21 / 2026-10-07.

## License

- Code, tests, data: MIT (see `LICENSE`).
- Manuscript (`paper/`): CC BY 4.0.
