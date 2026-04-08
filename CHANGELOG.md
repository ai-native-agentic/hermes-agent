# Changelog

This file tracks notable changes shipped on the
`ai-native-agentic/hermes-agent` fork. Upstream changes are pulled
periodically; the entries below are the additions on top of the
upstream sync point at commit `8b861b77`.

## Unreleased

### Added — Self-hosted provider support (`lunark`)

- **`lunark` provider** — first-class registration in
  `hermes_cli/providers.py`, `hermes_cli/auth.py`, and the `--provider`
  argparse choice list. Configurable via `LUNARK_API_KEY` and
  `LUNARK_BASE_URL` env vars. (`74e8622f`)

- **`/v1/models` capability flag passthrough** — when a custom
  OpenAI-compatible endpoint advertises `tool_calling` and `reasoning`
  booleans on each model, `agent/model_metadata.fetch_endpoint_model_metadata`
  now exposes them in the cache. (`74e8622f`)

- **Capability gating** — at agent boot, if the active model
  advertises `tool_calling: false`, drop all tool definitions before
  the run loop. Sending function definitions to a model that can't
  emit them is the root cause of "silent narration" failures. Backed
  by 4 unit tests. (`e35d124e`)

- **`acp_adapter/session.py` `model.max_tokens` propagation** — the
  ACP `_make_agent` previously ignored `model.max_tokens` from
  `config.yaml`, leaving reasoning models truncated mid-`<think>` chain.
  (`74e8622f`)

- **`examples/lunark/`** — 12 reusable runners for self-hosted vLLM
  validation: `provision.py`, `extractor.py`, `batch_qa.py`, `matrix.py`,
  `tool_calls.py`, `moa.py`, `moa_hard.py`, `moa_aggregator.py`,
  `variance.py`, `use_cases.py`, `advanced.py`, `multi_model.py`. Plus
  the shared `examples/acp_client.py` ACP JSON-RPC wrapper.
  (`cab921f5`, `baf82c7e`, `a8c570b8`, `1fcdd7e3`, `aec7f411`,
  `8a2aca74`, `5cae829d`)

- **`examples/lunark/README.md`** — quickstart, file map, and lessons
  learned from running every recipe against the live lunark cluster.

### Added — Self-learning safety net

- **Silent failure detection** (`run_agent.py`,
  `_maybe_warn_hallucinated_tool_action`) — when the model says
  "I created the skill" / "I saved that to memory" but no `tool_call`
  was actually emitted in the current turn, print a stderr warning
  with a copy-pasteable resume command. Best-effort heuristic that
  strips `<think>` blocks and never injects back into the conversation.
  Backed by 11 unit tests. (`e35d124e`, `38d281fd`)

- **`hermes insights` self-learning panel** — surfaces
  "skills created in the last N days" + "user-created total" + "most
  used (invocation count)" + "unused (>30d)" so users can see whether
  the procedural-memory loop is actually firing. (`e35d124e` + Q5)

- **Per-skill usage metrics** — `agent/skill_metrics.py` records
  `views` (skill_view calls) and `writes` (skill_manage successes)
  per skill into `~/.hermes/skills/.usage.json`, atomically. Surfaced
  via the insights panel above. Backed by 9 unit tests. (Q5)

### Added — Mixture-of-Agents flexibility

- **`tools/mixture_of_agents_tool.py` is now config-driven** — the
  built-in MoA tool was previously hard-wired to OpenRouter. A new
  `moa:` config block lets users point it at any OpenAI-compatible
  endpoint with custom reference models and aggregator. Backward
  compatible: when `moa:` is missing the OpenRouter defaults run
  unchanged. Backed by 7 unit tests. (`458ed2f5`)

### Added — Skill retrieval (Q1)

- **Top-k skill retrieval** — `agent/skill_retrieval.py` is a
  dependency-free TF-IDF ranker. `build_skills_system_prompt` now
  accepts `user_query` + `retrieval_config`; when
  `skills.retrieval.mode: topk`, only the top-k matching skills are
  injected into the system prompt instead of the full index. Bounds
  the prompt token cost at any skill-library size. Backed by 25
  unit + integration tests. (`c7216c88`)

### Added — Schema simplification

- **`skill_manage` description slimmed** — the original ~1500-token
  description in `tools/skill_manager_tool.py:SKILL_MANAGE_SCHEMA`
  exceeded smaller models' tool-selection budget. Slimmed to ~50
  tokens; full guidance moved to the parameter descriptions and
  `skill_view()` output. (Q6)

### Added — DX

- **`--provider` argparse choices auto-derived** —
  `hermes_cli/main.py` no longer hardcodes the provider list; choices
  are pulled from `PROVIDER_REGISTRY` + `HERMES_OVERLAYS` so adding a
  new provider doesn't require touching argparse. Surfaces 24
  providers (was 14). (Q2)

- **Persistent default profile root** —
  `examples/lunark/provision.py` now defaults to
  `~/.hermes/profiles/` instead of `/tmp/`, so HERMES_HOME profiles
  survive reboot. (Q7)

- **Lunark recipes CI gate** — new
  `.github/workflows/lunark-recipes.yml` runs the extractor self-test,
  the skill retrieval pytest, and a recipe import smoke test on every
  push that touches `examples/lunark/` or `agent/skill_retrieval.py`.
  (Q3)

### Added — Test coverage

- **78+ unit/integration tests added across the new code paths**:
  `tests/test_lunark_provider.py` (6),
  `tests/test_lunark_extractor.py` (48),
  `tests/test_hallucinated_tool_warning.py` (11),
  `tests/test_capability_gating.py` (4),
  `tests/test_skill_retrieval.py` (18),
  `tests/test_skill_retrieval_integration.py` (7),
  `tests/test_skill_metrics.py` (9),
  `tests/tools/test_mixture_of_agents_tool.py` (+4 new for the
  config-driven path).
