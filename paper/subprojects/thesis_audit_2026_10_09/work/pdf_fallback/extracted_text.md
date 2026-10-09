<!-- pdf_page: 1 -->


ConsensusCourt: Procedural Evidence Arbitration for
                             Scientific Claims
                  in Multi-Agent Research Pipelines

                                Wei Wang∗

                               September 21, 2026




                                       Abstract

         Multi-agent LLM research pipelines generate streams of evidence—retrieved literature, com-
      putational results, agent opinions—that frequently conflict.  Existing aggregation strategies
      (majority vote, Elo tournaments, single-judge selection) either ignore conflicts entirely or re-
      solve them by rhetorical weight, and both are vulnerable to two under-addressed failure modes:
       false convictions of well-formed claims built on low-quality evidence, and infrastructure fail-
      ures (citation-verification network outages) silently polluting scientific verdicts. We present
     ConsensusCourt, an arbitration framework that borrows its procedural guarantees from the
     courtroom rather than from voting theory: every claim is presumed innocent until a detected
      contradiction pair convicts it (an asymmetric burden of proof, ABP); evidence is weighted by an
       explicit source-pedigree, recency, and relevance scorer whose confidence channel is structurally
      decoupled from conviction; and an escalation layer routes uncertain or unadjudicable cases to
     human review instead of guessing. Three results carry the paper’s evidential weight and do not
     depend on benchmark authorship: (i) a counting law that calibrates the court’s advantage over
      guilty-presumption screening—the log-log slope of false pairs against pairwise opportunities is
      0.942 (expected 1.0) and the advantage-ratio slope is −0.857 (expected −1.0); (ii) a controlled
      network-failure experiment in which a three-state citation-verification policy holds the clean-
      claim false-positive rate at 0 under failure rates up to 100%, where the common two-state policy
      convicts up to 100% of innocent claims—the only variable being the failure-mapping policy over
     an otherwise identical verification stack; and (iii) a conviction–confidence decoupling invari-
      ant, preserved exactly under every scorer ablation. On the author-built synthetic layer of our
       four-layer 954-case benchmark (Ddb; evaluation at regression scale, §8), the full court reaches
       recall = 1.0 with FPR = 0.0 while majority vote, Elo calibration, and single-judge baselines
        fail on at least one side of the Pareto frontier at every threshold setting we scan. A per-case
      advantage decomposition identity holds exactly by construction, its empirical content carried
     by a pure layer attribution (residual = 0 on 148 negative-basis cases), and verdicts are order-
      invariant over 1,000 evidence permutations. The main benchmark and every red-line experiment
      are deterministic and zero-LLM; two registered LLM-side ablations (an LLM arbiter replacing
      the rule-based PI arbiter, and a PROClaim/DebateCV-style prosecution–skeptic–judge LLM
      debate protocol) are executed against a hosted LLM and reported in §6.12. All deterministic
      pipelines are guarded by pre-registered red-line assertions wired into a regression suite.

  ∗AI Senior Engineer, Shanghai Hongene Biologics Tech Co., Ltd. He currently leads AI architecture and ap-
plied R&D at a leading biopharmaceutical company, focusing on multi-agent AI applications in biomedical R&D.
Correspondence: weiransoft@126.com. Code, benchmark generators, experiment scripts, and all machine-readable
reports backing every number in this paper are available in the accompanying repository (backend/tests/ddb/ for
data, backend/tests/scripts/ for the deterministic experiment pipelines).



                                          1


<!-- pdf_page: 2 -->


1  Introduction

LLM-driven research agents now propose, execute, and report on entire scientific workflows.  In
drug-discovery pipelines, a single design decision may draw on dozens of evidence items produced by
heterogeneous agents: literature retrievals (with citations that must resolve), structure-prediction
outputs, simulation summaries, and free-text agent opinions. When two such items disagree—a
paper reports that an allosteric inhibitor is effective against a resistance mutation, another that
monotherapy fails in vivo—the pipeline must decide which claim survives, and it must do so procedu-
rally: retro-justified aggregation (“the loudest evidence wins”) is precisely how plausible-but-wrong
conclusions propagate.
   Two failure modes dominate. First, false conviction: systems that pre-screen evidence with a
guilty-presumption rule (“any malformed evidence rejects the case”) reject half of all well-formed
claims in our benchmark (FPR 0.5, Section 6.4), because the mere presence of a defect somewhere
in the evidence pool is treated as an indictment of the claim. Second, infrastructure pollution:
citation verifiers call network APIs, and when those APIs fail, a naive two-state policy (valid
vs. anything-else) classifies “could not verify” as “fabricated,” convicting clean claims at rates
that reach 1.0 under full network outage (Section 6.8). Neither failure mode is addressed in the
evidence-aggregation channel by the multi-agent debate literature, which studies argumentation
quality but not procedural burden of proof—even the recent courtroom- and debate-style verifiers
[1, 2] resolve conflicts through LLM rhetoric rather than an explicit, auditable conviction rule; nor
by peer-review conflict analysis [3, 4], which adjudicates conflicts among human reviewers rather
than among machine-generated evidence items inside a running pipeline.
  We contribute ConsensusCourt, an arbitration framework with four procedurally grounded
components:

  1. Evidence scoring with explicit pedigree (§3.2): every evidence item receives a source-
     pedigree,  recency, and relevance score, with weights 0.4/0.2/0.4 combined as overall =
     0.4 source + 0.2 recency + 0.4 relevance, with a registered weight table for 13 agent roles.

  2. Asymmetric burden of proof (ABP) (§3.3):  conviction requires a detected contradic-
     tion pair—two evidence items taking opposing stances on the same aspect—conditioned by
    aspect matching, an object-validity gate, and graded intensity; evidence is otherwise presumed
     innocent, and scoring is structurally decoupled from conviction.

  3. A seven-step court procedure (§3.4): symmetric pre-screening (hard constraints only),
     scoring, ABP adjudication, PI arbitration, independent review, human-BOSS override, and
     escalation, with six explicit escalation clauses.

  4. Three-state citation verification (§6.8): valid / invalid / network_failed, in which
     infrastructure failures carry evidentiary warning status and can never convict.

   Our evaluation rests on Ddb, a four-layer Dispute Benchmark (§5) spanning 954 cases: 600
synthetic L1 cases, 88 curated pseudo-conflict negatives (L1.5), 10 real literature disputes with ver-
ified PubMed identifiers (L2), and 256 intensity-graded cases (L4). The main benchmark and every
red-line experiment are deterministic, zero-LLM pipelines with pre-registered red-line assertions
executed by a CI-wired regression suite; two registered LLM-side ablations additionally invoke a
hosted LLM (§6.12). Every number in this paper is traceable to a machine-readable report.
   Contributions.  (i) A procedural arbitration framework whose conviction channel is struc-
turally decoupled from evidence scoring, with a theory-grounded advantage identity (§4); (ii) a



                                          2


<!-- pdf_page: 3 -->


four-layer benchmark with keyword-channel locking so that synthetic and real-dispute layers mea-
sure the same detection channel (§5); (iii) a controlled experiment on network-failure discrimina-
tion in citation verification—to our knowledge no prior work quantifies how verification outages
propagate into downstream verdicts:  industrial triage systems already deploy unexamined vari-
ants of the three-state policy [5], and our contribution is the controlled failure-rate/FPR curve
that measures the pollution cost of the ubiquitous two-state policy (§6.8); (iv) deterministic, red-
line-guarded, zero-LLM evaluation infrastructure in which component ablations, scorer ablations,
order-permutation robustness, source-label noise, and role-weight calibration are all reproducible
from a single regression script (§6); and (v) two executed LLM-side ablations—an LLM arbiter
replacing the rule core’s PI arbiter and a protocol-level LLM-debate verifier on identical evidence
pools—measured against the frozen court outputs under the same operating points (§6.12).

2  Related Work

Hallucination detection and self-verification. Surveys [6] catalogue hallucination modes in
generation; SelfCheckGPT [7] samples self-consistency signals to detect unsupported content, and
Reflexion [8] and deliberation methods [9] improve factuality by iterative verbal feedback. These
methods judge single outputs against internal consistency. ConsensusCourt addresses a comple-
mentary problem: when a pipeline’s evidence pool contains genuine, source-grounded disagreement,
internal consistency signals cannot resolve it; an explicit adversarial procedure must.
   Multi-agent LLM systems. Multi-agent debate improves reasoning by having models cri-
tique each other [10]; frameworks such as AutoGen [11] and MetaGPT [12] orchestrate role-
specialised agents; the AI co-scientist [13] coordinates hypothesis generation and review through
Elo tournaments. Critically, these systems’ final aggregation is either voting, tournament ranking,
or a single judge. Section 6.2 shows all three aggregation families fail on at least one side of the
recall/FPR frontier on Ddb, and Kapoor et al. [14] independently argue that agent evaluations are
dominated by cost- and rigor-insensitive methodology—our red-line-guarded deterministic harness
is a direct response.
   Machine-side courtroom and debate-based claim verification. Closest in spirit to Con-
sensusCourt are three recent systems that stage claim verification as an explicit adversarial
proceeding. PROClaim [1] casts verification as a courtroom debate between Plaintiff and Defense
agents before a Judge, with Progressive RAG expanding the evidence pool mid-debate and het-
erogeneous multi-judge aggregation enforcing calibration; DebateCV [2] (WWW 2026) pairs two
debater agents with a moderator adjudicator trained by Debate-SFT post-training; TRUST Agents
[15] combines a claim decomposer, a Delphi-style multi-agent jury, and a logic-aware aggregator for
fake-news verification. All three are LLM-driven, operate offline on standalone controversial claims,
and resolve conflicts by generated argumentation; ConsensusCourt is rule-driven, deterministic,
and zero-LLM, adjudicating conflicts among machine-generated evidence items at pipeline runtime
under an explicit burden of proof. The two approaches are complementary rather than competing:
a procedural court of the kind built here can wrap the outputs of an LLM debate as its evidence
pool, supplying the auditable conviction channel that stochastic argumentation lacks. We report a
protocol-level version of this comparison—an LLM debate verifier with the PROClaim/DebateCV
prosecution–skeptic–judge structure consuming identical evidence pools (§6.12); head-to-head eval-
uation against the third-party systems themselves remains future work (§8).
   Adjudication of review conflicts. Closest to us, Cecil and Freedman [4] (MDHJudgments)
study medical hallucination detection where annotator judgments conflict, and introduce dual-
track adjudication protocols for label curation; Kumar et al. [3] (IMPACT/RevCI) analyse conflicts



                                          3


<!-- pdf_page: 4 -->


among human peer reviews, proposing aspect-conditioned contradiction extraction and graded con-
flict intensity. Both adjudicate conflicts among human or labelling agents, offline, to curate datasets
or triage AC decisions. ConsensusCourt adjudicates conflicts among machine-generated evidence
items at pipeline runtime, where the decision is whether a research claim continues to execute. We
adopt their aspect-conditioning and graded-intensity insights as design components (§3.3), and our
dual-track escalation analysis (§3.4) mirrors their finding that adjudicators themselves diverge.
   Citation verification and infrastructure failures. Citation-verification services are increas-
ingly used as hallucination firewalls, and industrial systems have begun to implement three-way
triage—e.g., GPTZero’s hallucination check routes unverifiable citations to an explicit “unknown”
outcome rather than a fabrication verdict [5]—while abstention-aware evaluation is gaining ground
on the research side of scientific citation generation [16]. What no prior work we are aware of
quantifies, however, is how verification outages propagate into downstream verdicts: the industrial
triage practices ship without a failure-rate/FPR curve, and citation-generation abstention metrics
measure generation-side behaviour rather than downstream adjudication. Section 6.8 provides, to
our knowledge, the first controlled false-positive curve comparing failure-handling policies under
matched failure injection, showing the distinction is not implementation detail but a first-order
scientific-validity property.

3  The ConsensusCourt Framework

3.1  Design principles

Three principles, each falsifiable in our experiments:
  P1 (Presumption of innocence). A claim is rejected only when a detected contradiction
pair convicts it. Absence of contradictions acquits, regardless of evidence quality. This inverts the
guilty-presumption pre-screening common in retrieval firewalls, which we show costs 0.5 FPR on
clean claims (§6.4).
  P2 (Conviction–confidence decoupling). The scoring channel computes a confidence for
acquittal semantics; the conviction channel is driven solely by detected contradiction structure.
Any ablation of the scorer therefore cannot change recall/FPR—only calibrated confidence—an
invariant we assert as a red line (§6.6).
  P3 (Escalation over error). When the court cannot adjudicate— because evidence was
filtered, confidence is low, or tracks diverge—it abstains and escalates to humans. Abstention is a
correct answer, not a failure mode (§6.11, escalation-type case).

3.2  Evidence scoring with source pedigree

Each evidence item e carries a source key s(e) ∈{pubmed, chembl, uniprot, experiment,
agent_opinion, …}, an agent role r(e) from the 13-role registry (Appendix A), a publication
timestamp, and free-text content. The scorer produces

                overall(e) = 0.4 · source(e) + 0.2 · recency(e) + 0.4 · relevance(e),             (1)

where source combines the source key’s pedigree weight with the agent role’s registered weight,
recency decays with evidence age, and relevance scores topical fit to the claim under review. The
uniform-replacement value used in the source-pedigree ablation is the median of the eight registered
source weights, 0.775.




                                           4


<!-- pdf_page: 5 -->


3.3  Asymmetric burden of proof (ABP)

The rubric engine evaluates the case against a registry of contradiction keyword pairs (e.g., “effec-
tive/ineffective,” “promote/inhibit”), scoped by three mechanisms:

 • Aspect conditioning: pairs only match within a shared biomedical aspect (safety, eﬀicacy,
    mechanism, …), preventing cross-aspect pseudo-conflicts—the dominant failure mode predicted
    by Kumar et al. [3].

 • Object-validity gate: ambiguous or non-co-referent object mentions block pair formation
     (e.g., “drug X is toxic” vs. “drug Y is toxic” is not a contradiction about X).

 • Graded intensity: each detected pair receives an intensity grade 1–3 determined by polar-
      ity, source-pedigree differential, and same-object corroboration; intensity 3 (adversarial-grade)
     triggers mandatory escalation (clause 5, Appendix B).

Same-source evidence pairs are skipped (a source cannot contradict itself), and a hard-constraint
channel detects regulatory tool bans as single-evidence critical findings—e.g., references to Al-
phaFold2/3 structure prediction, which the deployment platform prohibits system-wide (both AF2
and AF3 are banned; compliant chains such as OpenDDE/Chai1/Boltz are whitelisted).  This
channel operates independently of keyword pairs and is the only conviction path that survives
ABP removal on synthetic data (§6.5). The ban covers AlphaFold2 [17] and AlphaFold3 [18].

3.4  The court procedure

hold_court executes seven steps: (1) symmetric pre-screening (hard constraints; e.g., tool bans);
(2) evidence scoring (Eq. 1); (3) ABP adjudication (contradiction pairs + intensity); (4) PI-agent
arbitration over the surviving evidence, with a decision of support/conditional support/reject by
weighted-average thresholds (≥0.7 / ≥0.4 / < 0.4); (5) independent review producing objections;
(6) human-BOSS override item generation; (7) escalation determination by six clauses (Appendix B),
including: any reviewer objection; any intensity-3 conviction; dual-track divergence (the rule track
and the evidence-review track disagree in direction); and no-adjudicable- evidence abstention. Our
headless experiment pipeline (run_court_headless) replicates this orchestration line by line and
is locked to the production hold_court by a field-exact parity test on an in-memory database; any
drift in production orchestration fails the suite.

4  Theoretical Guarantees

We state two structural propositions (set-algebra identities and implementation invariants—true
by construction rather than empirically falsifiable) and one falsifiable theorem (the counting law),
whose quantitative predictions drive the experimental design; full proofs and counting lemmas are
in the project’s outline document.

4.1  Setup

Let C = ∅denote the negative basis (claims with no true contradiction). Define: D: the guilty-
presumption pre-screen convicts (a defect exists in the pool); M: the ABP court convicts (a pseudo-
contradiction pair fires); S ⊆D \ M: defect-only cases (guilty screen fires, ABP does not); M\D:
ABP-only false convictions. Under C = ∅(no true contradiction), D and M events are by definition
false positives of their respective mechanisms.



                                           5


<!-- pdf_page: 6 -->


Table 1: Ddb layers. “Channel” names the detection channel the layer is designed to measure. All
layers are emitted by versioned generators and locked by consistency tests.

      Layer                             Size   Construction and target channel

     L1 (synthetic disputes)          600   Balanced with/without-contradiction cases from con-
                                                   trolled templates (300 defect-only, 150 clean-neutral
                                             within the negative half); measures ABP recall and
                              FPR on synthetic ground truth.
      L1.5 (pseudo-conflict negatives)  88    Three curated negative  families:   cross-aspect  (40),
                                            ambiguous-reference  (24),  different-object-gate  (24);
                                          measures gate purity (target FPR = 0).
     L2 (real literature disputes)      10     First batch  of  hand-built  disputes from  real pub-
                                                  lished literature; every evidence item carries verifiable
                                 PubMed identifiers,  all verified against NCBI esum-
                                     mary (re-verified 2026-09-20, with a second-pass correc-
                                                tion pass); gold contradiction pairs recorded; measures
                                                source-generalisation of ABP recall (H2b).
     L4 (intensity-graded)            256   16 templates × 4 rounds × 2 ∆-buckets × 2 gate buck-
                                                   ets with intensity gold labels from the S1–S5 decision
                                                   table; measures grading fidelity.


Proposition 1 (Per-case advantage decomposition). On the negative basis C = ∅,

                         Pr(D) −Pr(M)     = Pr(S) −Pr(M\D).                     (2)
                             |     {z     }
                           advantage of ABP over guilty screen

The advantage equals defect-only cases correctly spared minus pseudo-conflict cases wrongly con-
victed. This is a set-algebra identity that holds by construction; its empirical content is the layer
attribution measured in Section 6.9, not the closure itself.

Theorem 1 (Counting bound on the FPR ratio). Under a null case with m evidence items, single-
claim defect rate p0, per-pair pseudo-mismatch rate qK, and pairwise dependency κ (Section 6.9 for
the measured estimator), the expected FPRs satisfy
             FPRguilty ≈1 −(1 −p0)mm      ,    qK through K ∼κ q2K m   joint events,
         FPRABP      qK κ  2                             2
and the Bonferroni-consistent bound FPR ratio ≥1/(κ m   · qK ) tightens as m grows: the ABP                                                                 2     p0
advantage widens with case size, with crossover m∗where the counting-based budgets of the two
mechanisms intersect.

Proposition 2 (Order invariance). The canonical seven-step court produces a verdict invariant to
the input evidence permutation: the verdict signature (passed, convicted-pair multiset, pending pairs)
is identical for all orderings, because every aggregation is order-agnostic (global scoring, set-based
pair detection). Streamed early-termination variants forfeit this guarantee (Section 6.7).

5  The Dispute Benchmark (Ddb)

Ddb has four layers, all generated or curated deterministically from a single seed (42):
  H2b channel lock. Because L1 is generated by the same authors as the detector, generalisation
is a standing threat. On the first 10 L2 cases, the ABP keyword channel (direct rubric evaluation)


                                           6


<!-- pdf_page: 7 -->


recovers at least one gold contradiction pair in all 10 positives, and the L2 wording deliberately uses
the registry’s keyword pairs to express genuinely opposed stances—real positions that happen to hit
the keyword channel, never fabricated sentences. L2 output is exported to JSON deterministically
from the code single-source-of-truth with file↔code deep-equality locks. At n = 10 positives, recall
1.0 has a Wilson 95% interval of [0.72, 1.00]; the L2 result is therefore reported at hypothesis-
screening resolution only, and confirmatory interpretation awaits the planned L2 expansion (§8).
   Honesty boundary.  L2 ground truth  is expert-adjudication-pending; the gold pairs are
construction-time annotations, and the dual-track expert annotation protocol (three domain ex-
perts plus an independent evidence-verification track, with Krippendorff’s α reported even if low,
following 4) is specified but not yet executed (§8).

6  Experiments

6.1  Evaluation discipline

All red-line experiments are deterministic and zero-LLM: evidence scoring, pair detection, arbi-
tration, and escalation are pure code; no LLM is invoked in any of these pipelines (the production
platform’s LLM stack—vLLM-served Qwen [19]—is outside the scope of these experiments). Two
registered ablations carve out of this discipline and do invoke a hosted LLM—the LLM-arbiter ab-
lation and the protocol-level LLM debate of §6.12; they are excluded from the zero-LLM regression
and ship with their own structural red lines. Each experiment asserts pre-registered red lines
(e.g., “ablation FPR must not exceed full-court FPR”), violations abort with non-zero exit codes
wired into a regression suite; each check has a paired anti-test that verifies the checker itself catches
fabricated violations. Every report is machine-readable JSON; Appendix D lists commands, and
DATA_PROVENANCE.md maps every table in this paper to its report file.
  Why zero-LLM for the core experiments. The deterministic, zero-LLM discipline is a
methodological choice rather than a rejection of LLMs, for three reasons.  (1) Variable isolation:
the paper’s causal claim is that the ABP architecture controls false convictions, not that a partic-
ular model is capable. Running the core experiments without an LLM purifies the architectural
effect—any LLM component would confound attribution between “ABP works” and “the LLM is
good.” (2) Red-line verifiability: pre-registered assertions such as “ablation FPR must not exceed
full-court FPR” require deterministic outputs; stochastic LLM sampling would make red-line vio-
lations undecidable. (3) Reproducibility: every number in Tables 2–5 is reproducible from a single
regression script with no network or API key. The two LLM ablations of §6.12 deliberately break
this discipline to test the complementary question—what happens when an LLM arbitrates without
the ABP conviction rule—and the answer (FPR 0.500 for the single-LLM arbiter) confirms that
the architectural invariant, not model competence, is what bounds false convictions.
   Positioning within the agent-evaluation debate.  This harness is a deliberate answer
to the agent-evaluation critique: Kapoor et al. [14] show that agent benchmarks are dominated
by cost- and rigour-insensitive methodology; REDAgentBench [20] demonstrates that evaluation
verdicts can conflate actual violations with mere evidence visibility—a confusion isomorphic to the
infrastructure-failure pollution studied in §6.8; and AtomCite [21] shows that noise in automatic
(LLM-judge) annotation labels can bias measured verifier accuracy and reverse system rankings.
These findings motivate Ddb’s deterministic, versioned, code-locked construction: primary labels
are emitted by generators and locked by consistency tests, with expert adjudication planned as a
separate validation layer rather than the primary label source (§5, §8).




                                           7


<!-- pdf_page: 8 -->


Table 2: Main comparison on Ddb L1+L1.5 (n = 208, seed 42). “Reject” is the primary operating
point; “non-support” (reject or conditional) is a sensitivity operating point. Rejection-point Recall
and FPR carry Wilson score 95% confidence intervals. Best per column in bold.

                                         Reject operating point               Non-support sensitivity
System                  Recall   95% CI   FPR   95% CI    Acc   Prec  Recall FPR    Acc

ConsensusCourt (full) 1.000 [0.940, 1.000] 0.000 [0.000, 0.025] 1.000 1.000 1.000 0.000  1.000
 MajorityVote             0.550  [0.425, 0.669] 0.878  [0.816, 0.922] 0.245  0.203  0.883  0.993    0.260
 EloArbiter               0.050  [0.017, 0.137] 0.628  [0.548, 0.702] 0.279  0.031  1.000  1.000    0.289
 SingleJudge              0.000  [0.000, 0.060] 0.074  [0.042, 0.128] 0.659  0.000  0.833  0.939    0.284


6.2  Main comparison against aggregation baselines

Four systems consume the same EvidenceScorer scores on the same 208 cases (60 L1 positives +
148 negatives: 60 L1 + 88 L1.5); the only variable is the arbitration logic.  Baselines: Major-
ityVote (equal-weight agent votes, thresholds 0.7/0.4), EloArbiter (role-weighted average cali-
brated through a single-round Elo transform, approximating tournament review 13), SingleJudge
(max-confidence evidence decides alone).
   Three findings. (1) Constructive failure of vote-style aggregation: majority vote rejects
87.8% of negatives—because the negative half deliberately contains defect-bearing cases, and low
evidence quality drives rejection votes. (2) Single-judge zero recall: the highest-scoring evidence
on genuine disputes is frequently on the wrong side; the judge never sees the opposing item. (3)
Sensitivity collapse: under the “non-support” operating point, all three baselines push FPR to
0.94–1.0: without a contradiction channel there is no principled boundary between “conditional
support” and “reject.” The full court is the only system occupying recall = 1.0 / FPR = 0.0 on
both operating points. The Wilson intervals make the separation unambiguous rather than point-
estimate-lucky: the court’s recall interval [0.940, 1.000] and FPR interval [0.000, 0.025] overlap with
no baseline’s interval in either column.

6.3  Threshold-sensitivity scan

A fair objection to Table 2 is that baselines are evaluated at a single, possibly unfavourable, oper-
ating point (support threshold 0.7, reject threshold 0.4). We therefore scan the full threshold grid
(tsup, trej) ∈{0.3, . . . , 0.9}2—49 points—for each baseline, by decision-primitive replay: each case’s
scores and each baseline’s decision primitive (vote counts, Elo role-weighted average, single best
score) are computed once, and every grid point only remaps the primitive to a verdict. Production
code is untouched; at the default thresholds the replayed metrics are red-line-locked to Table 2
field-for-field. Figure 1 plots the resulting (FPR, recall) Pareto frontier of each baseline against
the full court. The empirical answer is unambiguous: no baseline threshold point reaches the
court profile (recall = 1.0 at FPR = 0.0). At FPR ≤0.034 no baseline attains positive recall at
all; the first positive-recall frontier point already costs FPR ≥0.43 (majority 0.432 at recall 0.033),
and recall at or above 0.467 (elo) is only reachable at FPR ≥0.946. The court’s advantage is thus a
property of its contradiction-channel architecture, not of a threshold choice made on the baselines’
behalf.

6.4  Pilot: ABP vs. guilty-presumption pre-screening

On the full 600-case L1 layer, the guilty-presumption pre-screen (any formal defect ⇒case rejection,
the standard retrieval-firewall policy) achieves recall 0.6 at FPR 0.5 overall—convicting 100% of


                                           8


<!-- pdf_page: 9 -->


MajorityVote
                                                        EloArbiter
                                                       SingleJudge
                                                              Full court (reference)


                             1



                                0.8



                                0.6
                                                           Recall
                                0.4



                                0.2



                             0
                                0           0.2          0.4          0.6          0.8         1
                                   FPR

Figure 1: Pareto frontiers of the three aggregation baselines over the 7×7 threshold grid (tsup, trej ∈
{0.3, . . . , 0.9}, reject operating point), from decision-primitive replay with zero production-code
changes. The star marks the full court, which is threshold-free by construction; no baseline grid
point attains recall = 1.0 at FPR = 0.0. Data: baseline_sensitivity_report.json.


defect-only negatives (by design) but also 50% of clean-neutral negatives via collateral pollution.
ABP holds recall 1.0 / FPR 0.0 on the same cases. This is the empirical form of Proposition 1: the
guilty screen’s P(D) = 0.203 on the 148-case negative basis is entirely defect-only (S), all of which
ABP spares (P(M) = 0).

6.5  Component ablations

ABP is load-bearing: without the pairwise rubric, conviction recall collapses 1.0 →0.0 (the
synthetic layer contains no tool-ban content, so the hard-constraint channel—the only remaining
conviction path—never fires). Reviewer and BOSS are downstream of conviction by design:
removing them leaves conviction identical (a structural invariant asserted by red line). BOSS is
the primary escalation signal source: removal drops escalation recall to 0.733 while conviction
is untouched.
   Escalation is not conviction. On the full court, escalation FPR is 1.0 with precision 0.5
(and 0.0 on the L1.5 basis): every defect-bearing-but-innocent case (the S set of Proposition 1) is
routed to human review rather than auto-rejected. This is the intended semantics—escalation is
the court’s abstention channel, and measuring precision@escalation is how the routing budget is
tuned (§6.11 shows the complementary escalation- type case where abstention is provably correct).





                                           9


<!-- pdf_page: 10 -->


Table 3: Court component ablation on 60×2 L1 (seed 42). Conviction layer vs. escalation layer.
no_abp removes the pairwise rubric (hard constraints remain); no_reviewer / no_boss remove the
downstream review/override components.

                                             Conviction     Escalation
                      Mode          Recall  FPR   Recall  FPR

                              court_full      1.000   0.000   1.000   1.000
                        no_abp       0.000   0.000   1.000   1.000
                            no_reviewer   1.000   0.000   1.000   1.000
                          no_boss       1.000   0.000   0.733   0.800


Table 4: Scorer-level ablation (60×2 L1, seed 42). avg_pass-conf = mean overall score of passed
cases. Recall/FPR identical across rows (decoupling invariant); the observable effect is confidence
inflation.
          Configuration                     avg. pass conf.             Interpretation

          full_v03 (reference)                   0.4994               calibrated baseline
          no_source_pedigree (s→0.775)      0.5248          low-pedigree sources lifted
          no_recency (→1.0)                  0.6250       stale evidence conflated with fresh


6.6  Scorer-level ablations

The conviction–confidence decoupling (P2) makes a scorer ablation’s effects observable only through
the confidence channel. We therefore run two scorer ablations with the full court’s switches frozen,
post-processing scores experiment-side (production code untouched), with two red-line layers: R1
(monotonicity: ablated FPR must not exceed and recall must not fall below full court) and R2
(effect direction: no_recency confidence must not decrease; no_source_pedigree confidence must
differ from the original—proving the ablation is live).
   Both red lines hold: recall/FPR are exactly 1.0/0.0 in all rows, and the confidence ordering
quantifies each mechanism’s contribution to calibration (+0.025 from source pedigree, +0.126 from
recency—the largest single effect, showing recency decay is doing most of the work in discounting
stale literature).

6.7  Robustness: order, size, streaming, and label noise

Order invariance (Proposition 2). 1,000 random permutations across 60 L1 positives, both full-
court and baseline configurations: 0 divergent verdict signatures (invariant rate 1.0). The court’s
verdicts are a function of the evidence set, not its order.
   Streaming ablation. A rule-based streamed variant that terminates on first contradiction
(k=1) or first k-strong-streak (k=2) diverges from the canonical court in 70/1000 = 0.070 permu-
tations (k=1; 10/60 cases affected; permuted recall 0.924 vs. canonical 0.95, court canonical 1.0)
and 2/1000 = 0.002 (k=2, recall 0.998).  Order sensitivity is real for early-termination designs
and decays as the termination threshold tightens—supporting the design choice of full deliberation
before disposition.
   Counting-law calibration (Theorem 1). Under controlled null cases (p0 = 0.30 designed
defect injection, unique-source construction so that all m  pairs are live, 300 trials per m ∈                                                                   2
{3, 5, 10, 20}): the log-log slope of false pairs against m   is 0.942 (expected 1.0), the advantage-                                                               2
ratio slope against (m−1) is −0.857 (expected −1.0), and the empirical ratio meets the Bonferroni



                                          10


<!-- pdf_page: 11 -->


Table 5: Network-failure experiment (Fig. 2): FPR on clean claims / recall on invalid-reference
claims, by policy and failure rate p.

                                             p = 0   p = 0.3  p = 0.5  p = 1.0

                     Three-state FPR (clean)    0.000   0.000   0.000   0.000
                    Two-state FPR (clean)       0.000    0.667    0.833    1.000
                     Three-state recall (invalid)   1.000    1.000    0.900    0.000†
                    Two-state recall (invalid)     1.000    1.000    1.000    1.000

 †At full outage the three-state court abstains on all cases (refuses to adjudicate rather than guessing)—consistent
   with the presumption-of-innocence principle; the two-state policy’s “perfect recall” is purchased entirely with
                                             clean-claim false positives.


bound pointwise at every grid point; the crossover is m∗≈18.5. The court’s advantage widens with
case size exactly as the counting law predicts.
   Source-label noise (±20%). A corruption pass re-labels exactly max(1, ⌊0.2m⌉) sources per
case (50 replicates per case; only source fields change, with field-level oracle invariants asserted).
On 7,400 negative replicates (L1 + L1.5), zero false convictions are induced—the only theoretical
attack (un-ignoring a same-source skipped pair) has no fuel on these bases, and the red line treats
this claim as falsifiable: any induced case would be written to an audit field with its pair endpoints
(observed: empty). On 3,000 positive replicates, 62 convictions are lost (2.07%; the sole channel is
a core pair re-labelled same-source), leaving noisy recall 0.9793; pair-set flip rate 13.9%, intensity
flip 14.1%, mean |∆conf| 0.0076 (max 0.578, attained when all pairs are simultaneously suppressed).
The positive-loss asymmetry is reported, not red-lined: it is a directional finding, not a defect.

6.8  Network-failure discrimination

Citation verifiers call network APIs; verification outages are routine. The community-standard
two-state policy (valid vs. “anything else”) conflates “could not verify” with “fabricated.” Our
production three-state policy maps network_failed to a warning that cannot convict, with 3×
exponential-backoff retries and a cross-case TTL cache. We inject deterministic transport failures
(httpx.MockTransport) at rates p ∈{0, 0.3, 0.5, 1.0} over 60 clean-claim cases (all references valid,
shared 40-hot-PMID pool) and 20 invalid-reference cases (recall control). Both policies consume the
identical production verification/retry/cache stack; the only variable is the failure-mapping policy.
    Process metrics confirm the mechanisms: the three-state policy’s mean attempts rise 1.00 →
1.33 →1.75 →3.00 with p (retry effect observable), and the cross-case cache absorbs 133 lookups
(shared-pool reuse), cutting transport requests at p = 0.3 from 212 to 103. Six pre-registered red
lines pass, including bit-exact determinism of a double-run at p = 0.5.
   Relation to abstention-aware evaluation. The three-state policy instantiates the same
design stance as abstention-aware evaluation in scientific citation generation, where hallucination-
rate-centred metrics dominate [16]: uncertainty should yield an explicit non-committal outcome, not
a conviction. The contribution here is orthogonal—citation-generation abstention metrics measure
generation-side behaviour, whereas Table 5 quantifies the downstream adjudication cost of omitting
the third state, as a function of the transport failure rate.

6.9  Advantage-decomposition calibration

On the 148-case negative basis (C = ∅; 60 L1 negatives of which 30 defect-only, plus 88 L1.5
pseudo-conflicts), with the real pre-screen as the D labeler and the real ABP rubric as the M


                                          11


<!-- pdf_page: 12 -->


1        two-state direct
                                               three-state resilient                                                        claims
                         0.75
                                               clean  0.5
                   on
                         0.25
                            FPR
                        0
                          0            0.3      0.5                  1
                                        transport failure rate p


Figure 2: Clean-claim false-positive rate under injected citation-verification failures. The two-state
policy’s FPR is monotone in p and reaches 1.0 at full outage; the three-state policy is structurally
flat at 0 (network failures cannot enter the conviction channel).  Data: Table 5; source report
tests/ddb/network_failure_report.json.


Table 6: Proposition 1 calibration. Cell counts and closure of Pr(D) −Pr(M) = Pr(S) −Pr(M\D)
on the negative basis.

                     Config       P(D)   P(M)   Pr(S)  Pr(M\D)   residual

                     full_v03       0.2027   0.0000   0.2027    0.0000      0
                     baseline_v02   0.2027   0.5946   0.2027    0.5946      0


labeler:
   The identity holds by construction (residual 0, tolerance 10−12, for both configurations—this is
expected, since the identity is a set-algebra proposition, not an empirical finding); the empirical
content is the pure layer attribution:  all S cases (30/30) come from L1 defect-only;  all M\D
(88/88 under baseline switches) come from L1.5 pseudo-conflicts—each benchmark family provably
measures exactly the channel it was built for.
   Dependency calibration. On controlled null cases (m = 20, 300 trials, 57,000 pair opportuni-
ties per configuration): baseline qK = 0.0355 vs. full qK = 0.0058 (the gates cut per-pair mismatch
rates 6–13×); adjacent-pair joint events yield κ = 1.42 (baseline) and 2.47 (full), both above the
κ ≥1 independence line (stronger dependence in the full court is expected—gates induce correlated
suppression), with joint-event counts (1,829 / 86) adequate for estimation.

6.10  Role-weight calibration

The 13 agent roles’ evidence weights (Appendix A) were calibrated by single-role perturbation over
a 208-case pool (L1 60×2 + L1.5), for both arbitration arbiters that consume role weights. Red
lines: coverage ≥5 cases/role (achieved 27–55), positive control (×0.1 perturbation) must flip ≥1
decision (achieved 7/6/6 flips on pi-agent/drug-researcher/report-expert—proving the pipeline is
live), and decision flip rate ≤0.05 at ±20%.
   Observed maximum flip rate: 4/208 = 0.0192 (data-specialist at ×0.8, identical for both ar-
biters), with flips occurring only in the down-weight direction and far below the 0.05 line. Verdicts
are driven by contradiction structure, not by weight tuning: the role table is a calibration surface,
not a decision surface.




                                          12


<!-- pdf_page: 13 -->


6.11  End-to-end case studies on real literature disputes

Two L2 disputes, run through the full seven-step court with full intermediate-state logging (timeline,
ABP pair ledger with intensities, per-evidence scores, filtered-evidence audit, verdict chain):
   Deadlock type (DDB-L2-0001): EGFR triple-mutation inhibition. Both sides carry
top-venue evidence (allosteric route: Nature 2016 [22] / Cancer Lett 2017; ATP-covalent route:
NEJM FLAURA/OS [23]). Zero evidence is filtered; ABP detects the gold contradiction pair (L2-
01-EV-02 × L2-01-EV-03, intensity 3), the case is convicted, and the PI arbitrates conditional
support at confidence 0.528; four escalation clauses fire (low confidence, conditional-support policy,
reviewer objection, intensity-3 mandatory review). This is the “both sides are credible” regime in
which vote-style aggregation degenerates: the court produces a traceable chain from detected pair
to human-review routing.
   Escalation type (DDB-L2-0005): AF3 complex-prediction dispute. Every evidence
item cites AlphaFold2/3, hitting the platform’s structure-prediction tool ban; the symmetric pre-
screen filters all evidence; the court finds itself with no adjudicable evidence, abstains (decision
unable to adjudicate, confidence 0.0), and escalates the entire case to human review.  Gold-pair
detection is not applicable in this regime—the gap between the two cases’ regimes is itself a finding:
the court distinguishes “conflict to resolve” from “nothing to judge,” and refuses to fabricate a
verdict in the latter.
   Honesty note. Grounded in verified real literature, these case studies are not production-
platform runs; the outline’s envisioned “5-round group-chat deadlock” narrative segment is replaced,
faithfully, by the seven-step court timeline, and production-run case studies (ASFV and canine
vaccine pipelines, §8) remain pending until the L3 layer is built.

6.12  Executed LLM-side ablations

The two ablations registered as pending in the research plan—swapping the rule-based PI arbiter
for a hosted LLM, and running a PROClaim/DebateCV-style prosecution–skeptic–judge debate
verifier on identical evidence pools—have now been executed against a hosted LLM (qwen-flash,
temperature 0; configuration and commands in Appendix D). Both are evaluated on the same
208-case pool as Table 2 (60 positives, 148 negatives), with the court’s outputs frozen in advance
from the pre-registered baseline report, under the identical reject and non-support operating points.
Evidence is presented blind: evidence_id, source, role, content only—no scores, no gold labels,
no court verdicts. Table 7 reports the outcome, taken verbatim from llm_arbiter_report.json
and llm_debate_report.json.
   Three observations, reported regardless of direction. (1) Recall is not what separates the
rule core from LLM arbitration. The single-LLM arbiter reaches 0.983 recall—within one
positive case of the court’s 1.0—because on this synthetic benchmark outright contradictions are
linguistically conspicuous. (2) The separation is entirely in the false-conviction dimension.
The LLM arbiter convicts FPR = 0.500 [0.421, 0.580] of the 148 well-formed negatives while the
court convicts none (FPR 0.000, CI upper bound 0.025); on the non-support operating point the
arbiter convicts 0.980 of negatives (accuracy 0.303). The asymmetric burden of proof—not rhetor-
ical competence—is what keeps benign evidence out of the conviction column, which is precisely
the paper’s central claim measured on the LLM’s home turf. (3) Adding adversarial structure
without a conviction rule does not help. The 3-call debate protocol degrades relative to single-
shot arbitration (recall 0.700 vs. 0.983; ∆recall = −0.300 vs. the frozen court) and never supports a
well-formed claim (0 support verdicts out of 208—consistent with the prosecution’s framing prevail-
ing over the skeptic’s across both polarities, as judged by the final verdict distribution), while still



                                          13


<!-- pdf_page: 14 -->


System              Recall ↑    95% CI   FPR ↓    95% CI

                Court (frozen reference, rule-based PI arbiter)
              ConsensusCourt    1.000    [0.940, 1.000]   0.000    [0.000, 0.025]

            LLM arbiter (208 cases, 0 unresolved failures)
           LLM arbiter          0.983    [0.911, 0.997]   0.500    [0.421, 0.580]

            LLM debate (208 cases × 3 roles, 0 unresolved failures)
           LLM debate          0.700    [0.575, 0.801]   0.372    [0.298, 0.452]

Table 7: Executed LLM-side ablations on the 208-case pool (reject operating point; positive class
= case contains a true contradiction). The court row is frozen from the pre-registered baseline
comparison; the two LLM rows are executed results with resume-for-missing-calls (each case retains
its first successful response; single model, one protocol round per case). Sensitivity (non-support)
operating points, per-verdict distributions, and call-failure accounting are in the accompanying
JSON reports.


convicting 0.372 of negatives. Structured LLM argumentation inherits the same false-conviction
failure mode that motivates the court’s presumption of innocence, and the debate framing adds no
mechanism to bound it.
   Honesty note.  These runs are exploratory rather than tuned: one hosted model at tem-
perature 0, one round each, no prompt ensembling. A pre-registered configuration deviation is
declared: the planned max_tokens=300 was raised to 2048 because the hosted qwen-flash gate-
way enables chain-of-thought by default and the 300-token budget was exhausted by reasoning
tokens, leaving the verdict payload empty; the deviation and its reason are recorded in the reports’
max_tokens_deviation fields. Structural red lines all hold (complete response coverage, verdict-
enum integrity with raw-response re-parsing, frozen court reference match); conservative handling
of unresolved cases (not-caught under reject, caught under non-support) is declared in the reports’
prediction_mapping fields. Results follow the same pre-registration discipline as the deterministic
experiments and are excluded from the zero-LLM regression.

7  Discussion

Why courtroom procedure, not voting. The baseline failures in Table 2 are not implementa-
tion artefacts but category errors: majority voting aggregates preferences, while scientific arbitra-
tion must aggregate evidence under a burden of proof. Once conviction requires a contradiction pair,
the system’s recall/FPR profile is governed by detection quality (gates, aspects, intensity) rather
than by vote thresholds—and every detection component is individually ablatable (Tables 3–4)
with red-line-guarded attribution.
   Escalation as a first-class outcome. The escalation layer’s apparently poor “precision”
(0.5 on smoke sets, 0.0 on the negative basis) is the design working as intended:  the S set of
Proposition 1—defect-bearing, contradiction-free, therefore innocent—must be routed to humans
rather than auto-rejected. The operational question is routing budget, not escalation FPR; the
deadlock / escalation case-study pair (§6.11) illustrates both regimes.
  An escalation cost model. The routing question can be made falsifiable.  Let c be the
cost of one human review, L the loss incurred by one wrongful automatic rejection of an innocent
claim, Nesc the number of escalated cases, and ˆp the precision@escalation. The escalate-over-error
policy costs Cesc = c Nesc; an auto-reject policy that dismisses exactly the escalated set costs



                                          14


<!-- pdf_page: 15 -->


Cauto = (1 −ˆp) L Nesc. Escalation is the cheaper policy if and only if c/L < 1 −ˆp. On the 148-case
negative basis (§6.9), every escalated case is innocent (ˆp = 0), so escalation dominates for any
c < L: one wrongful auto-rejection must cost more than one review. On the 60×2 L1 smoke set
(§6.5), ˆp = 0.5, so escalation dominates iff c < L/2. The model turns the P3 principle from a design
stance into a testable proposition: any deployment reporting precision@escalation and review cost
can read off the cheaper channel, and the red-line-guarded precision measurements (§6.5, §6.9)
supply the measured inputs.
   Infrastructure failures are a scientific-validity concern. Table 5 shows that a standard
engineering default (fail closed) translates directly into convicting up to 100% of clean claims during
outages. A three-state evidentiary policy costs nothing at p = 0, preserves most recall at moderate p
through retries and caching, and abstains rather than fabricates at full outage. We suggest that any
verifier-gated adjudication system report its failure-mode FPR curve alongside its clean-condition
accuracy.
  ABP and LLMs are complementary, not competing. A common misreading of the zero-
LLM discipline is that ConsensusCourt rejects LLM integration. The opposite is true: ABP
is a procedural architecture structurally decoupled from the arbitration substrate, and LLMs can
occupy any of three slots inside it.  (i) As the scorer: an LLM can replace the rule-based source-
pedigree/relevance scorer (Eq. 1); by the P2 decoupling invariant this changes confidence calibration
but cannot move recall or FPR. (ii) As the contradiction detector: an LLM can enrich the aspect-
matching and intensity-grading that feed ABP conviction, potentially raising recall on semantically
non-obvious disputes.  (iii) As the arbiter: an LLM can replace the PI-agent in step 4 of §3.4;
§6.12 tests this slot without ABP and finds FPR 0.500, which is precisely why the ABP conviction
rule must be retained regardless of arbiter choice. The target production architecture is therefore
LLM components inside an ABP envelope, not LLM-as-verifier without procedural guardrails; the
zero-LLM core experiments isolate the envelope’s contribution, and the LLM ablations quantify
what happens when the envelope is removed.

8  Limitations

We state the boundaries of the present evidence explicitly.

  1. No user study yet. The planned 10-participant blind A/B evaluation (SUS trust, NASA-
   TLX workload, open feedback vs. Elo and single-judge outputs) has not been run; all claims
     here concern system behaviour, not user experience.

  2. No production-run (L3) layer. Ddb’s L3 layer—claims, evidence, and verdicts harvested
    from real platform runs, including the group-chat deadlock narratives and wet-lab action
    orders—does not exist yet.  The case studies (§6.11) use verified real literature but are
    benchmark-grounded, not production logs.

  3. LLM-side ablations:  partial. Two of the registered LLM-side ablations have now been
    executed against a hosted LLM (qwen-flash, temperature 0): the LLM-arbiter ablation (swap-
    ping the rule-based PI arbiter for a direct LLM three-verdict arbiter) and a protocol-level
    LLM-debate verifier following the PROClaim [1] and DebateCV [2] prosecution–skeptic–judge
     structure, both consuming identical evidence pools with the court’s outputs frozen in advance
    and evaluation under the same reject/non-support operating points of Table 2 (§6.12). The
     third registered ablation—replacing the position-weight proxy with LLM-based structural role
    annotation—remains pending, as does head-to-head evaluation against the third-party verifier



                                          15


<!-- pdf_page: 16 -->


systems themselves; the protocol-level comparison reported here uses our own debate prompts
    rather than the systems’ trained pipelines.

  4. L2 scale and annotation status.  The L2 layer is 10 cases (target 45+),  all positives
    (FPR-relevant neutral sets are a second-batch item), and expert adjudication is pending; gold
     pairs are construction-time annotations, and keyword-channel sensitivity means ABP’s L2
    conclusions must eventually be cross-validated with expert and LLM-judge channels. The
    expansion protocol is already designed: grow L2 to at least 45 cases including a neutral negative
    family (target FPR = 0 under the same gate-purity red line as L1.5), have every case annotated
    by three domain experts under the dual-track protocol (three experts plus an independent
     evidence-verification track), and report Krippendorff’s α even  if low, following 4; the pre-
     registered acceptance red line is that ABP recall on expert-adjudicated L2 must stay within
    the Wilson interval reported in §5. Results below that bound would falsify the generalisation
    claim and will be reported as such.

  5. Third-party benchmark cross-validation. All current benchmark layers are author-built.
   The next cycle adds a cross-validation arm on at least one independently constructed claim-
     verification benchmark, with the court and the strongest baseline evaluated under identical
     splits, prompts (where applicable), and operating points; divergence between author-built and
     third-party results will be analysed and reported regardless of direction. This arm is designed
    but not yet executed, and no third-party numbers appear in this paper.

  6. Smoke-scale statistics. Reported experiments run at regression scale (60–300 cases per cell)
     for determinism and CI cost; the pre-registered main experiment (600×2 full-combinatorial
     ablations with significance tests) runs on the same pipelines at scale.

  7. Zero-LLM scope.  The main benchmark,  all baselines, and all red-line experiments are
     deterministic rule systems; the two LLM-side ablations of §6.12 are the only pipelines that
    invoke an LLM, and the production platform’s LLM components (evidence authoring, group-
    chat debate) remain upstream and unevaluated here.

9  Conclusion

ConsensusCourt reframes evidence aggregation in multi-agent research pipelines as procedural ar-
bitration: presume innocence, convict only on detected contradiction structure, score for calibrated
confidence, escalate what cannot be judged, and never let infrastructure failures impersonate scien-
tific findings. Three results carry the evidential weight and do not depend on benchmark authorship:
the counting law’s calibrated advantage (slopes 0.942/−0.857 against expected 1.0/−1.0, with the
empirical ratio meeting the Bonferroni bound at every grid point), the three-state verification pol-
icy’s zero network-failure false positives under matched injection, and the conviction–confidence
decoupling invariant preserved under every scorer ablation. On the author-built synthetic layer of
the 954-case benchmark (evaluation at regression scale, §8), the full court is the only system that
holds recall 1.0 / FPR 0.0 against all three standard aggregation baselines at every threshold setting
scanned, and its verdicts are permutation-invariant. Every number from the deterministic pipelines
is produced under pre-registered red lines wired into CI, and the two executed LLM-side ablations
report their hosted-LLM results against the same frozen operating points (§6.12). The path to a
complete preprint runs through the pending items above: expansion and expert adjudication of
L2, third-party benchmark cross-validation and third-party head-to-head verifier evaluation, the
L3 production layer, the user study, and the LLM-based structural role annotation ablation—each
is a bounded, scripted increment on infrastructure that already exists.


                                          16


<!-- pdf_page: 17 -->


References

 [1] Masnun Nuha Chowdhury, Nusrat Jahan Beg, Umme Hunny Khan, Syed Rifat Raiyan,
   Md Kamrul Hasan, and Hasan Mahmud. Courtroom-style multi-agent debate with progressive
   RAG and role-switching for controversial claim verification. arXiv preprint arXiv:2603.28488,
    2026.

 [2] Haorui He, Yupeng Li, Dacheng Wen, Yang Chen, Reynold Cheng, Donglong Chen, and
     Francis C. M. Lau.  Debating truth:  Debate-driven claim verification with multiple large
    language model agents.  In Proceedings of the ACM Web Conference 2026 (WWW), 2026.
    arXiv:2507.19090, DOI 10.1145/3774904.3792993.

 [3] Sandeep Kumar, Yash Kamdar, Abid Hossain, Bharti Kumari, Tanik Saikh, and Asif Ekbal.
   When reviews disagree: Fine-grained contradiction analysis in scientific peer reviews. Findings
     of the Association for Computational Linguistics: ACL 2026, 2026. arXiv:2605.10171.

 [4] Joe Cecil and Marjorie Freedman. Beyond majority vote: Multi-perspective adjudication for
    medical hallucination detection. Findings of the Association for Computational Linguistics:
   EMNLP 2026, 2026. arXiv:2609.03953.

 [5] GPTZero.      How   does   GPTZero’s    hallucination   check    detector   work?
   A     technical     report.            Technical     report,     https://gptzero.me/news/
    how-does-gptzeros-hallucination-check-detector-work-a-technical-report/, 2026.

 [6] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
    Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation.
   ACM Computing Surveys, 55(12):1–38, 2023.

 [7] Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. SelfCheckGPT: Zero-resource black-
    box hallucination detection for generative large language models.  In Proceedings of the
    2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.
    arXiv:2303.08896.

 [8] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and
    Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in
    Neural Information Processing Systems (NeurIPS), 2023. arXiv:2303.11366.

 [9] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griﬀiths, Yuan Cao, and Karthik
    Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In
    Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2305.10601.

[10] Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving
     factuality and reasoning in language models through multiagent debate. In Proceedings of the
    40th International Conference on Machine Learning (ICML), 2023. arXiv:2305.14325.

[11] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun
    Zhang, Shaokun Zhang, Jiale Liu, et al. AutoGen: Enabling next-gen LLM applications via
     multi-agent conversation. arXiv preprint arXiv:2308.08155, 2023.

[12] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang,
     Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. MetaGPT: Meta program-
    ming for a multi-agent collaborative framework.  In International Conference on Learning
    Representations (ICLR), 2024. arXiv:2308.00352.


                                          17


<!-- pdf_page: 18 -->


[13] Juraj Gottweis et al. Towards an AI co-scientist. arXiv preprint arXiv:2502.13164, 2025.

[14] Sayash Kapoor, Benedikt Stroebl, Zachary S. Doggle, Nitya Ghosh, and Arvind Narayanan.
    AI agents that matter. arXiv preprint arXiv:2407.01502, 2024.

[15] Gautama Shastry Bulusu Venkata, Santhosh Kakarla, Maheedhar Omtri Mohan, and Aish-
    warya Gaddam. TRUST agents: A collaborative multi-agent framework for fake news detection,
     explainable verification, and logic-aware claim reasoning.  arXiv preprint arXiv:2604.12184,
    2026.

[16] Yash Saxena, Deepa Tilwani,  Ali Mohammadi, Edward  Raff, Amit Sheth,  Srinivasan
    Parthasarathy, and Manas Gaur.  Attribution in scientific literature: New benchmark and
    methods. arXiv preprint arXiv:2405.02228, 2024.

[17] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ron-
     neberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al.
    Highly accurate protein structure prediction with AlphaFold. Nature, 596:583–589, 2021.

[18] Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf
    Ronneberger, Lindsay Willmore, Andrew Ballard, Joshua Bambrick, et al. Accurate structure
     prediction of biomolecular interactions with AlphaFold 3. Nature, 630:493–500, 2024.

[19] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph
    Gonzalez, Hao Zhang, and Ion Stoica. Eﬀicient memory management for large language model
     serving with PagedAttention.  In Proceedings of the 29th Symposium on Operating Systems
     Principles (SOSP), 2023.

[20] Zixing Chen, Xingyuan Liu, Jie Zhu, Huaixia Dou, Shuo Jiang, Junhui Li, Lifan Guo, Feng
    Chen, and Chi Zhang. REDAgentBench: Executable red teaming and faithful measurement
     of LLM agent systems. arXiv preprint arXiv:2608.10669, 2026.

[21] Chen Qian, Yimeng Wang, Yu Chen, Lingfei Wu, and Andreas Stathopoulos.  AtomCite:
     Verification and correction of supplied page-level citations in multi-page documents.  arXiv
     preprint arXiv:2609.05802, 2026.

[22] Yun Jia, Cai-Hong Yun, Eunhee Park, Dalia Ercan, Manisha Manik, Juergen Hartjoana, Rong
    Guan, Carla Liang, and Pasi A. Janne. Overcoming EGFR(t790m) and EGFR(c797s) resis-
    tance with mutant-selective allosteric inhibitors. Nature, 534:129–132, 2016. PMID: 27251290.

[23] Jean-Charles Soria, Yuichiro Ohe, Johan Vansteenkiste, Thanyanan Reungwetwattana, Bun-
    cha Chewaskulyong, Ki Hyeong Lee, Arthid Dechaphunkul, Fumio Imamura, Nobuyuki
    Nogami, Takayasu Kurata, et al. Osimertinib in untreated EGFR-mutated advanced non-small-
      cell lung cancer. New England Journal of Medicine, 378(2):113–125, 2018. PMID: 29151359.


A  Agent role evidence-weight registry

Evidence-credibility weights used by both arbiters (registered per role; unregistered roles fall back
to 0.5; the platform’s data-analyst implements the paper’s data-specialist profile, with the
platform-side fallback difference disclosed in the project’s role-profile appendix):
   Design principles: first-hand evidence > computational heuristics > organisational/expressive
cues; adjacent gap ≥0.05.


                                          18


<!-- pdf_page: 19 -->


Role                  Weight  Role               Weight

                      pi-agent                  1.0       data-specialist        0.75
                     drug-researcher          0.9       qc-inspector          0.7
                       structural-biologist       0.9      knowledge-engineer   0.65
                    computational-chemist   0.85      reflexion-agent        0.6
                     synthetic-chemist        0.8       project-coordinator   0.6
                     antibody-engineer        0.8       report-expert         0.5
                     peptide-designer         0.75


B  Escalation clauses

The escalation decision is the disjunction of: (1) any reviewer objection; (2) PI confidence below
the conditional-support threshold; (3) dual-track divergence between the rule-arbitration track and
the evidence-review track; (4) decision in a policy-mandated review category; (5) any intensity-
3 convicted contradiction pair (mandatory human review); (6) insuﬀicient adjudicable evidence
(abstention). Clauses 4 and 6 are what make the escalation-type case study (§6.11) behave as it
does: a fully filtered case cannot be convicted or acquitted, so it must be escalated.

C  Pre-registered red lines (summary)

Each experiment family ships with machine-checked red lines, each paired with an anti-test that
fabricates a violation to prove the checker is live: baseline comparison (court strictly dominates
all three baselines on both operating points); component ablation (no_abp conviction collapse on
L1; reviewer/BOSS conviction invariance; BOSS escalation monotonicity; headless pipeline parity
with production hold_court); scorer ablation (R1 monotonicity; R2 effect direction); robustness
(permutation divergence = 0; counting-law slopes within bands; noise-induced negative convictions
= 0 with audited empty set; noise recall ≥0.95; mean |∆conf| ≤0.05); advantage decomposition
(exact closure; zero critical evidence; P(S) > P(M\D); qK ordering; κ ≥1); network failure (three-
state FPR ≡0; recall profile; two-state monotone pollution curve; two-state recall invariance;
bit-exact double-run; cache effect); role calibration (coverage, positive control, flip rate).

D  Reproducibility

All experiments run from a single regression entry point, the bash script backend/tests/scripts/
run_v03_regression.sh, with deterministic seed 42 and no network or LLM calls (the network-
failure experiment’s failures are injected at the transport layer). The two LLM-side ablations of
§6.12 run outside this regression via a dedicated entry point and require network access to a hosted
LLM endpoint. Per-experiment commands and report files (last two rows: network required):
   The last two rows require network access to a hosted LLM (config via LLM_EXPERIMENT_*
environment variables; model qwen-flash, temperature 0) and are excluded from the zero-LLM
regression.
  Ddb layer files: tests/ddb/{l1,l15,l2,l4}_cases.json, each locked to its generator by deep-
equality consistency tests.





                                          19


<!-- pdf_page: 20 -->


Experiment                  Script                                 Report

Main comparison           run_baseline_comparison.py            baseline_comparison_report.json
Threshold sensitivity        run_baseline_sensitivity.py           baseline_sensitivity_report.json
Component ablation        run_ablation_v03.py --court           ablation_v03_report.json
Scorer ablation             run_ablation_v03.py --scorer-ablation ablation_v03_report.json
Robustness A–D            run_robustness_experiments.py         robustness_report.json
Decomposition             run_advantage_decomposition.py        advantage_decomposition_report.json
Role calibration            run_role_weight_calibration.py        role_weight_calibration.json
Network failure             run_network_failure_experiment.py     network_failure_report.json
Case studies               run_case_study.py                     case_study_report.json
H2a pilot                  pilot scripts (tests/ddb)             pilot_h2a_report.json
LLM arbiter ablation       run_llm_arbiter_experiment.py         llm_arbiter_report.json
LLM debate (protocol-level) run_llm_debate_experiment.py          llm_debate_report.json





                                      20
