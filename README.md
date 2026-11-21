<p align="center">
  <a href="https://research.samaya.ai/benchmarks/frontier-finance">
    <img src="assets/samaya_logo.png" alt="Samaya AI" width="220">
  </a>
</p>

<h1 align="center">Samaya AI's FrontierFinance Benchmark Grader</h1>

<p align="center">
  <a href="https://research.samaya.ai/benchmarks/frontier-finance"><img src="https://img.shields.io/badge/Benchmark-research.samaya.ai-4c6ef5?logo=readthedocs&logoColor=white" alt="Benchmark"></a>
  <a href="https://huggingface.co/datasets/samaya-ai/FrontierFinance"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-samaya--ai%2FFrontierFinance-ffce3a" alt="Dataset"></a>
</p>

Rubrics-based LLM grader for Samaya AI's FrontierFinance benchmark.
Given a set of **rubrics** per query and the **system responses** you want to grade, 
an LLM judge panel decides whether each rubric is satisfied, and the tool reports 
qualification-rate metrics.

It is self-contained: it talks to provider APIs through their official SDKs and
has no dependency on any internal evaluation framework.

Read more about FrontierFinance and our system benchmarking results in [our blog post](https://samaya.ai/blog/frontier-finance).

![Rubrics qualification rate vs. cost per query across systems, with the Samaya System leading the desirable high-quality/low-cost region.](assets/pareto_graph.png)

## Install

The judge backends are optional extras ("flavors") — install only the providers
you'll use:

```bash
uv pip install 'frontier-finance[anthropic]'          # Claude judges (default)
uv pip install 'frontier-finance[openai,gemini]'      # mix and match
uv pip install 'frontier-finance[all]'                # all three backends
```

Extras: `anthropic` (claude-* models), `openai` (gpt-* / o* models), `gemini`
(gemini-* models), `all`. The core install pulls in only PyYAML; configuring a
judge model whose backend isn't installed raises an error telling you which
flavor to add.

For local development on this package:

```bash
uv sync --extra dev   # installs all backends + pytest
```

## Get the data

The Hugging Face dataset ships the **rubrics** (`frontier_finance_public.jsonl`);
you bring your own system responses to grade against them. Download the rubrics
into the current directory with:

```bash
./scripts/download_data.sh          # -> ./frontier_finance_public.jsonl
./scripts/download_data.sh some/dir # or into a directory of your choice
```

The dataset is public — no login or token needed. The script uses the `hf` CLI
when available and otherwise falls back to `curl`.

The FrontierFinance rubrics dataset is published separately under the
[Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)
license; see its [dataset card](https://huggingface.co/datasets/samaya-ai/FrontierFinance).
Using it requires attribution per those terms.

## Run

### Step 1 — Export the API key(s) for your judge models

Set the key for whichever judge models your config uses:

```bash
export ANTHROPIC_API_KEY=...   # claude-* judges (default)
export OPENAI_API_KEY=...      # gpt-* / o* judges
export GEMINI_API_KEY=...      # gemini-* judges
```

### Step 2 — Write your run config

Create a YAML config pointing at your rubrics and responses. See
[`eval.example.yaml`](eval.example.yaml) — the exact config we used to produce
our reported numbers — and adjust the paths, `judge_models`, and knobs as needed:

```yaml
rubrics_path: frontier_finance_public.jsonl   # from ./scripts/download_data.sh
responses_path: system_summaries.json         # your own system responses
output_dir: grader_results/

judge_models:
  - claude-sonnet-4-6
  - gemini-3.1-pro-preview
  - gpt-5.4

max_rubrics_per_call: 30   # rubrics per LLM call
concurrency: 8             # queries graded in parallel
max_tokens: 64000          # max output tokens per judge call
```

**Note on judge models**: We select the three default judge models so that they achieve high agreement with a committee of 9 individual judges. Therfore we use these three judges across all our experiments.
Using three judges from three different model providers also help alleviate bias caused by any single model family.
**For standardized grading and fair comparison of results, we recommend that you use the same three judges (Sonnet 4.6, Gemini 3.1 Pro, GPT 5.4).**


### Step 3 — Run the grader

```bash
uv run frontier-finance-grader --config eval.example.yaml
```

Writes `metrics.json` and `per_item.json` to the config's `output_dir` and logs
a metrics table.

## Input formats

**Rubrics JSONL** — what you get from `download_data.sh`, for reference; one query
per line:

```json
{
  "query_id": "85bc71fb95e1a5c4",
  "query": "What are Databricks' main competitors?",
  "query_date": "2024-10-31",
  "rubrics": [
    {
      "rubric_id": 1,
      "rubric_text": "Indicates that Databricks is a private company...",
      "must_have": false,
      "rubric_type": "Qualitative & Contextual Information",
      "data_source_type": "company_originated_content"
    }
  ]
}
```

**Responses JSON** — the responses from your own system, as a JSON array; each
`query_id` must match a rubrics line:

```json
[{"query_id": "85bc71fb95e1a5c4", "system_summary": "Databricks competes with..."}]
```

## How judging works

For each query, the rubrics are sent to each judge model (in batches of
`max_rubrics_per_call`) with the query, its date, and the response. The judge
returns a per-rubric boolean. With multiple `judge_models`, each rubric is
decided by **majority vote** (the first model breaks ties). Use an **odd** panel
so a majority always exists — a tie is only possible with an even number of
surviving votes.

**Judge failures.** A judge that errors (or returns unparseable JSON through all
retries) is dropped from the vote, and the rubric checks it skipped are counted
against it in `failed_criteria_checks_by_judge`. As long as at least one judge
survives, the query is still graded on the surviving votes — a partial judge
failure is *not* a failed query. When **every** judge fails on a present response
(`failure_reason: "judge_error"`), that's a grader-side failure, not the
system's, so the query is **excluded from every scored metric** (`num_records`
and all denominators) — equivalent to skipping it — and only reported via
`num_judge_errors` and the per-judge stats. A query with no response
(`failure_reason: "no_response"`) is a system failure and *does* count as failed.

## Metrics

A rubric is *qualified* when the response satisfies it. *Macro* averages mean
per-query rates; *micro* averages total qualified / total rubrics. *Success*
queries are those that were graded; *failed* queries contribute 0 qualified and
their full rubric count toward the "all queries" denominators. *must_have* is the
essential rubric subset.

`metrics.json` includes:

- `num_records` (excludes judge errors), `num_failed_queries` (no-response failures), `num_judge_errors`, `num_rubrics_on_success_queries`, `num_rubrics_on_all_queries`, `num_qualified_rubrics`
- `failed_criteria_checks_by_judge` — per judge model, how many rubric checks it failed to produce across the run
- `macro_avg_qualification_rate_on_success_queries`, `micro_…`
- `…_on_all_queries`
- `…_must_have_…` variants of all four
- `breakdown_by_rubric_type`, `breakdown_by_data_source_type` (micro rate per taxonomy value, over success queries)

`per_item.json` has the per-rubric judge decisions for each query, plus each
query's `failure_reason` and `failed_checks_by_judge`.

## Unit Tests (No API keys needed)

```bash
uv run pytest
```

## Citation

If you use FrontierFinance dataset or this grading code, please cite as the following:
```
@article{zhang2026frontierfinance,
  title   = {FrontierFinance: A Challenging Benchmark for Measuring Frontier Intelligence of Finance Agents},
  author  = {Zhang, Yuhao and Koyluoglu, O. Ozan and Venkatesh, Thejas and Diehl Martinez, Richard and Bhatia, Vishank and Alidoust, Arash and Paranjape, Ashwin},
  year    = {2026},
  url     = {https://samaya.ai/blog/frontier-finance}
}
```

## License

This grading code is licensed under the [Apache License 2.0](LICENSE).

The FrontierFinance rubrics dataset it operates on is licensed separately under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) and is not covered by
this repository's license. See [NOTICE](NOTICE) for details.

## Harbor parity (fork only)

This fork adds `harbor_parity/` for the parity experiment of the
[Harbor FrontierFinance adapter](https://github.com/harbor-index/adapters/tree/main/src/frontier-finance).
The grader is upstream `8b71342` plus one compatibility change in
`frontier_finance/judges.py` for newer judge models: no `temperature` for Claude
Haiku 5+ and GPT-6+, which reject it, and `LOW` thinking when a Gemini 3 Flash
model rejects `MINIMAL`. Prompts, batching, parsing and voting are unchanged, and
the upstream panel gets the same requests as before. Upstream ships no agent, so the original
side runs `claude-code` outside Harbor with the same version, model, flags,
environment variables, instruction text and time limit as Harbor's
`claude-code` agent.

- `harbor_parity/run_claude_code.py run` reads the Harbor-generated tasks and runs one
  fresh container per query: local Docker, a Modal sandbox (`--backend modal`) or a
  Daytona sandbox (`--backend daytona`, needs `daytona<2` and `DAYTONA_API_KEY`).
  The image is the task's own `environment/Dockerfile` plus Harbor's `claude-code`
  install step. Each run gets the task's CPU, memory and time limits and the
  instruction on stdin, and keeps `/app/report.md`, the `stream-json` log and the
  session files. Rerunning skips queries with a final status; a backend failure
  (`backend_error`) is not final and is retried.
- `harbor_parity/run_claude_code.py collect` applies one rerun rule to an original run
  or a Harbor job (plus its rerun jobs) and writes the summaries for the grader.
  On a Harbor job it also lists trials whose grading lost a judge's votes, by the
  same rule as `score`, and leaves them out of the mean reward until regraded.
- `harbor_parity/eval.parity.yaml` is the grader config for parity: the subset's
  rubrics only, and the parity judge panel
  (`claude-haiku-5-5`, `gemini-3.8-flash`, `gpt-6-luna`), the same as
  `FF_JUDGE_MODELS` on the Harbor side. These judges are not the upstream panel, so parity scores
  are not comparable with the upstream leaderboard.
- `harbor_parity/run_claude_code.py score` merges the grader's `per_item.json` files
  into `macro_avg_qualification_rate_on_all_queries` and lists queries that lost a
  judge's votes, for a regrade.

```bash
export ANTHROPIC_API_KEY=... ANTHROPIC_BASE_URL=...   # the parity gateway
python harbor_parity/run_claude_code.py run -d <adapters>/datasets/frontier-finance-parity \
    -o runs/original1 -m claude-haiku-5-5 --claude-code-version 2.1.326 -n 8 \
    --backend daytona
python harbor_parity/run_claude_code.py collect runs/original1 \
    --summaries runs/original1/system_summaries.json
# once: the subset's rubrics, so the macro is over the 11 queries, not all 220
cat <adapters>/datasets/frontier-finance-parity/*/tests/rubrics.jsonl > parity_rubrics.jsonl
# judge keys and base URLs: ANTHROPIC_*, OPENAI_*, GEMINI_API_KEY, GOOGLE_GEMINI_BASE_URL
uv run frontier-finance-grader --config harbor_parity/eval.parity.yaml \
    --responses runs/original1/system_summaries.json --output-dir runs/original1/grades
python harbor_parity/run_claude_code.py score runs/original1/grades/per_item.json
```

The Harbor side runs `run_frontier-finance_parity.yaml` from the adapter with
the same model and version. `collect jobs/<job> --summaries ...` exports its
reports so they can be cross-graded here.
