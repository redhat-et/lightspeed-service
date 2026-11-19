# CPU Ask-Mode Quality Test Specification

This document specifies how to measure and compare answer quality when OLS ask-mode
traffic is served by CPU-hosted models, and how to evaluate the configuration changes
this branch ships:

- `ols_config.embed_model` — remote embedding backends for RAG (vLLM pooling runner or TEI)
- `solr_hybrid.pre_retrieval` — retrieve documentation before the first LLM call

The writer-swap arm in §4 needs no code: the writer is chosen per request with
`provider`/`model`, so pointing a query at a GPU or externally-hosted model is a
configuration change, not a branch difference.

It is a methodology spec, not a results report. Run it on your own hardware and quote
your own numbers.

## 1. Objective

Answer two questions with paired, statistically honest measurements:

1. Does a CPU-hosted model answer ask-mode questions acceptably, compared to a
   GPU-hosted or externally-hosted writer?
2. What is the measured effect of each configuration change above, individually?

## 2. Datasets

- **Full set:** `eval/eval_data.yaml` — 797 OpenShift questions, ask-mode shaped.
- **Evaluation slice:** a fixed subset of 100 questions, selected once and then frozen.
  Record the `eval_data.yaml` commit hash with every result so the slice is reproducible.
- **Smoke set:** `eval/eval_data_short.yaml` — 10 questions for wiring checks only.
  **No pass/fail claims may be made from the smoke set.**
- Do not modify `eval/eval_data.yaml` during a comparison campaign. Corrections to
  answer keys must be recorded as separate override files and applied identically to
  every arm at scoring time.

### Gold-key audit (do this before trusting any score)

In a measured audit of one 100-question slice, several percent of answer keys were
wrong or contradicted by the very corpus the system retrieves from — a retired product
name, an invented UI path, a claim the documentation contradicts. A wrong key does not
just mis-score one question; it actively rewards wrong behavior.

Before comparing arms, audit every failing question against (a) the retrieved chunks
recorded in the run artifact, (b) live probes against the documentation index, and
(c) neighbouring keys in the same file. Classify each failure as: gold defect,
unanswerable-as-written question, retrieval miss, or answer-side failure. Only
answer-side failures measure the model.

## 3. The ruler: a multi-judge panel is mandatory

**Do not score arms with a single judge.** Measured on identical answers, a single
judge pass at temperature 0 moved 24 of 95 per-question scores, with swings of
−0.20/+0.10, flipping one question on noise alone. The practical consequences:

- single-judge deltas below ~5 PASS (at n=100) are indistinguishable from noise;
- per-question inter-judge spread has a median around 0.10;
- only roughly half of questions are unanimous across judges.

**Panel protocol:**

| Element | Requirement |
|---|---|
| Judges | 3 independent LLM judges from different model families |
| Independence | No judge may be the same model as an arm under evaluation where avoidable; if unavoidable, report per-judge breakdowns so self-preference is visible |
| Scoring | Average the judges' per-question scores; PASS threshold 0.75 |
| Missing scores | A judge call that returns no score is recorded, not silently dropped |
| Fixed ruler | Freeze the judge set, threshold and gold overrides for the whole campaign; changing any of them mid-campaign invalidates cross-arm comparisons |

Refer to judges in shared artifacts by role (judge A/B/C) or public model name. Never
include provider endpoints or API details in anything shared publicly.

## 4. Arms

Run arms as pairs against a fixed baseline. Minimum matrix for the features on this
branch:

| Arm | Configuration |
|---|---|
| 0. Baseline | Stock pipeline, docs tool round, bundled embedder |
| 1. Pre-retrieval | `solr_hybrid.pre_retrieval: true` |
| 2. Remote embedder | `embed_model` pointing at a vLLM pooling-runner service |
| 3. Both | 1 + 2 together |
| 4. Writer swap | Same pipeline, strong writer model (GPU or external) |

Additional comparisons worth running when model choice is the question: a small model
on GPU for ask mode (the strongest skeptic's alternative), and an all-CPU arm to locate
the quality floor.

## 5. Statistics

- **Paired comparison, always.** Both arms answer the same questions; compare
  per-question outcomes, not just aggregate pass counts.
- **McNemar's test** on the discordant pairs (questions one arm passes and the other
  fails). Report +rescued/−regressed counts, the p-value, and n.
- **Pre-register gates before running.** State the minimum net gain, the maximum
  tolerated regressions on currently-passing questions, and the significance level.
  A gate decided after seeing results is not a gate.
- **Power awareness.** At n=100, a net gain of ~5 PASS is generally not detectable
  (a McNemar test with ~20 discordant pairs needs hundreds of paired questions for
  80% power at that effect size). If an arm matters, either accept a larger effect
  size or move to the full 797-question set.
- **Report per-question diffs** for every paired comparison: which questions flipped,
  in which direction, and the judge consensus on each.

## 6. Serving setup

- **Inference:** vLLM for both CPU and GPU pools; CPU images on CPU nodes.
- **Qwen models:** serve with `enable_thinking: false` (via
  `--default-chat-template-kwargs`). With thinking enabled and a reasoning-token cap,
  Qwen can spend the entire budget on reasoning and return empty content — this
  presents as an OLS bug but is an engine configuration issue.
- **Embedder:** serve the embedding model behind an OpenAI-compatible
  `/v1/embeddings` endpoint (vLLM pooling runner). The embedding model must match the
  checkpoint the documentation index was built with.
- **Model candidates for CPU ask mode:**
  - Primary: a ~30B MoE instruct model with ~3B active parameters (e.g. Qwen3-30B-A3B-Instruct-2507)
  - Lightweight fallback: an 8B-class instruct model (e.g. Granite 4.2 8B)
  - Larger dense CPU models (100B+) measured no better than the 30B MoE at several
    times the cost; do not assume scale helps on CPU.

## 7. Known pitfalls (measured, do not re-litigate without new evidence)

- **Review-time verification hurts.** Having a second model check, correct, or rewrite
  a finished answer was measured in three configurations (weaker draft verified by a
  stronger model; self-verification; cross-model verification) and never improved the
  result — the best case was a wash, the worst a large regression. Quality is decided
  at generation time. Measure with judges; never give a judge write access to the
  answer.
- **Retrieval knobs are usually not the lever.** Varying `max_results`,
  `max_expansion_neighbors` and vector boost returned byte-identical passages, because
  the per-chunk token budget and the corpus chunking structure — not the knobs —
  bound what is retrieved. Diagnose with a retrieval-only probe (no LLM calls) before
  spending an eval arm. If chunks have no sibling sections in the index, no
  configuration can retrieve them; that is a re-chunking problem.
- **Strict grounding prompts can be net negative.** A prompt that insists the model
  answer only from retrieved context makes retrieval gaps visible as omissions — and
  the judge, comparing against a key written from the full page, docks them. Measure
  any prompt change with the panel and per-question diffs.
- **Aggregate scores are stable, per-question scores are not.** A judge's total can
  reproduce across runs while individual scores swing. Trust deltas in aggregates
  over stories about single questions, except where the judge consensus is unanimous.

## 8. What to report

For each arm: pass count, mean score, median and p95 execution latency, and the
provider/model that served each round. For each pair: McNemar result, rescued and
regressed question lists with judge consensus. For the campaign: dataset versions
(commit hashes), judge set, gold overrides applied, and the frozen slice definition.
