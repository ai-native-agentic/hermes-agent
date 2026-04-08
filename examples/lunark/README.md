# Lunark + Hermes ACP examples

End-to-end recipes for running Hermes Agent against a self-hosted lunark vLLM
gateway over ACP. Built and tested against the live `https://llm.lunark.ai/v1`
endpoint.

## Quick start

```bash
# 1. Install Hermes with the [acp] extra
uv tool install --editable ".[acp]"
# (or: uv tool install --editable . --with agent-client-protocol --with mcp)

# 2. Auto-provision one HERMES_HOME profile per lunark model
python examples/lunark/provision.py
#    → writes /tmp/hermes_<MODEL>/{config.yaml,.env} for each model in /v1/models

# 3. Smoke test the Qwen3-32B profile
HERMES_HOME=/tmp/hermes_Qwen3-32B hermes chat -q "What is 17*23?" -Q

# 4. Run the 100-prompt baseline against one model
python examples/lunark/batch_qa.py
#    → /tmp/acp_results.jsonl

# 5. Run the 4-model × 100-prompt matrix
python examples/lunark/matrix.py
#    → /tmp/acp_matrix_results.jsonl
```

## Files

| File | What it does | Output |
|---|---|---|
| `provision.py` | Reads `<base_url>/models`, generates an isolated `HERMES_HOME` profile per model. Idempotent — rerun whenever the lunark lineup changes. | `/tmp/hermes_<MODEL>/` |
| `batch_qa.py` | 100 single-turn QA prompts across 10 categories (math, logic, knowledge, code, translate, format, text, stepwise, compare, creative) with a normalizer that handles LaTeX/markdown. Single model. | `/tmp/acp_results.jsonl` |
| `matrix.py` | Cartesian product: 4 lunark models × 100 prompts = 400 ACP calls via 8-worker thread pool. | `/tmp/acp_matrix_results.jsonl` |
| `tool_calls.py` | 10 tool-using scenarios (`terminal`, `read_file`, `execute_code`) per model. Verifies whether `tool_calling: true` in `/v1/models` is real. | `/tmp/acp_tools_results.jsonl` |
| `moa.py` | Mixture-of-Agents demo: query all 4 models in parallel, majority-vote the extracted answer, compare against each single model. | `/tmp/acp_moa_results.json` |
| `advanced.py` | ACP advanced features: multi-turn coherence, `session/list`, `session/fork` divergence, `session/cancel` mid-stream. Raw JSON-RPC. | `/tmp/acp_advanced_results.json` |
| `multi_model.py` | 5 representative prompts × 4 parallel models, fast smoke comparison. | `/tmp/multi_model_results.json` |
| `extractor.py` | Robust answer extraction (LaTeX/markdown/fraction/decimal/word normalization) used by the verifier-style runners. | — |
| `moa_hard.py` | 100 hard arithmetic / logic / trivia prompts × 4 models, both simple and weighted majority voting. Persists per-job JSONL for post-hoc analysis. | `/tmp/acp_moa_v2_jobs.jsonl` |
| `moa_aggregator.py` | Paper-style MoA: 3 reference models in parallel → 1 aggregator (Qwen3-32B) synthesizes. | `/tmp/acp_moa_aggregator_results.json` |
| `variance.py` | 5-run variance measurement on the 100 hard prompts (2,000 calls). Reports per-model mean ± stdev and stable vs flippy prompts. | `/tmp/acp_variance_summary.json` |
| `use_cases.py` | 50 practical use cases across 10 industry categories (code / devops / data / research / automation / writing / learning / math / meta / assistant), each routed to the most appropriate model based on V3 variance data. | `/tmp/uc50_results.jsonl` |

All runners depend on `examples/acp_client.py` (the reusable
`HermesACPClient` wrapper) and assume `hermes` is on `PATH`.

## Lessons learned (data from running these against lunark, April 2026)

- **Single-turn QA**: All four lunark models hit ≥95 % on the 100-prompt
  baseline once `model.max_tokens` is propagated through ACP (see commit
  `74e8622f`). `Qwen3-32B` and `Qwen3.5-27B` reached 100/100.
- **Tool calling is uneven**: `tool_calling: true` in `/v1/models` is a
  *capability advertisement*, not a guarantee. Real autonomous tool use
  ranged from 10/10 (`Qwen3-32B`) to 0/10 (`Qwen3.5-27B`) on the same
  prompts. Always verify with `tool_calls.py` before relying on it.
- **MoA = downside protection**: Majority voting across all four models
  recovered 5 individual model errors on stepwise reasoning, lifting the
  weakest model (Gemma 73 %) to 100 % at the cost of ~37 s wall time per
  15-prompt batch. It does not exceed the best single model when the best
  model is already 100 %.
- **`HERMES_HOME` isolation works**: Four `hermes acp` processes can run
  concurrently against the same lunark gateway with no DB lock or session
  collisions. We sustained 8 worker threads / 400 calls in ~4 min wall.
- **ACP advanced features pass cleanly**: `session/fork` deep-copies
  history and lets branches diverge independently; `session/cancel` is a
  notification (no response) and a long prompt stops within ~1 s with
  `stopReason: 'cancelled'`.

## Provisioning a different lunark deployment

```bash
LUNARK_API_KEY=mykey \
python examples/lunark/provision.py \
  --base https://my-llm.example.com/v1 \
  --root /var/lib/hermes-profiles \
  --json
```

## Adding a new model to the matrix

After running `provision.py`, edit the `MODELS` list at the top of
`matrix.py` / `moa.py` / `multi_model.py` to include the new id. The
profiles already exist on disk so no further config work is needed.
