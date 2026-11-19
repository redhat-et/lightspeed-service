# Reproducing the measured ask-mode result

This directory pins the ruler and the slice used for the published numbers, so a
reader can reproduce them on their own hardware instead of quoting ours.

Two results are published from these branches:

| branch | pipeline | result | mean score |
|---|---|---|---|
| `gold-cpu-ask-mode` | pre-retrieval, CPU model writes | **65 / 100** | 0.741 |
| `gold-gpu-ask-mode` | CPU runs the tool round, configured `synthesis_llm` writes | **70 / 100** | 0.763 |

Both are scored on the first 100 conversations of `eval/eval_data.yaml`
(`slice-100.txt`), with three judges averaged per question and PASS at 0.75.

## What you need

1. An image built from this branch (the config keys in the spec are only read by
   this code), running with the model pool below.
2. The OpenShift documentation index in Solr and an embedding endpoint serving the
   checkpoint the index was built with. If the embedder and the index disagree the
   vectors are silently wrong and every number here is meaningless.
3. A judge endpoint reachable over an OpenAI-compatible API, with three models.
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

`render_config.py` is deliberately dumb: it substitutes `${VAR}` from the
environment, refuses to run with a missing variable, and never writes a secret
into the file (the key is passed through `environment: OPENAI_API_KEY`, which the
harness exports before use). Keep `judges.env` out of git.

## Reading the result

At n = 100 a pass rate near 0.70 has a 95% interval of roughly ±9 points, and the
panel moves individual questions too. Reproducing means landing within that band
with the same *shape* of failure — not hitting 65 or 70 exactly.

Compare question by question, not only on the total: a run that scores 64 with the
same 30 failures is a reproduction; a run that scores 64 with a different 30 is a
different pipeline.

## Why three judges

On identical answers a single judge moved 24 of 95 per-question scores and flipped
one question on noise alone, and per-question inter-judge spread has a median near
0.10. A single-judge delta below ~5 PASS at n = 100 is not distinguishable from
noise. The template ships three judges from different families and averages them;
do not collapse it to one to save cost, or your numbers are not comparable with
these.
