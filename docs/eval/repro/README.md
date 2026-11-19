# Reproducing the measured ask-mode result

This directory pins the ruler and the slice used for the published numbers, so a
reader can reproduce them on their own hardware instead of quoting ours.

**This branch reproduces the CPU arm: 65 / 100.** The GPU arm (82 / 99) lives on
`gold-gpu-ask-mode`; the code is identical, only the writer the request is routed
to differs.

The two published results:

| branch | pipeline | result | mean score |
|---|---|---|---|
| `gold-cpu-ask-mode` | pre-retrieval, a CPU-hosted model writes every round | **65 / 100** | 0.741 |
| `gold-gpu-ask-mode` | pre-retrieval, the request is routed to a GPU model that writes the whole answer | **82 / 99** | 0.837 |

Both are scored on the first 100 conversations of `eval/eval_data.yaml`
(`slice-100.txt`), with three judges averaged per question, PASS at 0.75, and the
seven gold corrections described below. The GPU arm scored 99 because `conv_019`
returned an empty response in that run and produced no judgeable turn; on a run
where it succeeds the denominator is 100.

Neither arm uses `ols_config.synthesis_llm`. The GPU result is a whole-request
writer: the same model generates every round, it is just reached over a different
`provider`/`model` in the query.

## What you need

1. An image built from this branch (the config keys in the spec are only read by
   this code), running with the model pool below.
2. The OpenShift documentation index in Solr and an embedding endpoint serving the
   checkpoint the index was built with (`granite-embedding-small-english-r2` for
   the published runs). If the embedder and the index disagree the vectors are
   silently wrong and every number here is meaningless.
3. A provider entry in `olsconfig.yaml` for the writer. For the CPU arm it is your
   CPU model; for the GPU arm it is an `openai`-type provider pointed at your GPU
   endpoint. The harness selects it per request, so the app default can stay on
   the CPU model.
4. A judge endpoint reachable over an OpenAI-compatible API, with three models.
   You supply the endpoint, the key and the model names — nothing is hardcoded.

## Run it

```bash
cd docs/eval/repro
cp judges.env.example judges.env        # fill in endpoint, key, model names
export $(grep -v '^#' judges.env | xargs)   # or use your own secret injection
python3 render_config.py                 # writes system_panel3.yaml
```

Then score the slice. `lightspeed-eval` lives in the evaluation repository, not in
this one:

```bash
lightspeed-eval \
  --system-config docs/eval/repro/system_panel3.yaml \
  --eval-data eval/eval_data.yaml \
  --conv-ids $(cat docs/eval/repro/slice-100.txt) \
  --output-dir <your-output-dir>
```

`render_config.py` is deliberately dumb: it substitutes placeholders from the
environment, refuses to run with a missing variable, and never writes a secret
into the file (the key is passed through `environment: OPENAI_API_KEY`, which the
harness exports before use). Keep `judges.env` out of git.

## The answer key

The published numbers are scored against the upstream gold with seven
corrections, all of which are committed on this branch: `conv_017`, `conv_021`,
`conv_044`, `conv_051`, `conv_056`, `conv_097`, `conv_099`. They replace outdated
procedures (Service Catalog, pre-ODF images, removed CLI flags) with what the
documentation corpus actually says.

Scoring those questions against the uncorrected upstream gold gives a different,
lower number that is not comparable with the published table. Do not "fix" the
threshold or the key to match a run; use the key on the branch.

## Reading the result

At n = 100 a pass rate near 0.70 has a 95% interval of roughly ±9 points, and the
panel moves individual questions too. Reproducing means landing within that band
with the same *shape* of failure — not hitting 65 or 82 exactly.

Compare question by question, not only on the total: a run that scores one or two
below the published number with the same failures is a reproduction; a run that
scores the same with a different failure set is a different pipeline.

## Why three judges

On identical answers a single judge moved 24 of 95 per-question scores and flipped
one question on noise alone, and per-question inter-judge spread has a median near
0.10. A single-judge delta below ~5 PASS at n = 100 is not distinguishable from
noise. The template ships three judges from different families and averages them;
do not collapse it to one to save cost, or your numbers are not comparable with
these.
