# Local LLM Ryoku Support Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make Ryoku Query a local, source-grounded Discord support bot that responds normally inside one configured support channel, and only when mentioned or replied to elsewhere, using Gemma 4 E4B for concise answers and LFM 2.5 for bounded diagnostic reasoning.

**Architecture:** Keep safety classification, catalog retrieval, invocation policy, and citation selection deterministic. The LLM is a final-answer renderer that sees only the user's question and curated/Prowl evidence; it never decides whether destructive actions are allowed and it never receives or emits visible chain-of-thought. Ollama runs one model at a time with `keep_alive=0`, because the 14 GiB host cannot safely retain Gemma's 9.6 GB model and LFM's 5.2 GB model together.

**Tech Stack:** Python 3.12, discord.py, Ollama `/api/chat`, existing model2vec retrieval, Prowl Agent CLI, unittest.

---

### Task 1: Add a bounded Ollama client

**Objective:** Create a testable async local-LLM boundary that supports both installed models without exposing thinking text.

**Files:**
- Create: `src/agent/ollama.py`
- Test: `tests/test_ollama.py`

**Step 1: Write failing tests**
- Gemma requests use `think: false` and return only a trimmed final answer.
- LFM requests use `think: true`, use the returned `message.content`, and discard `message.thinking`.
- Responses that contain `<think>...</think>` are stripped before Discord output.
- Invalid JSON, non-200 responses, missing content, and timeout return a typed unavailable result.
- The client limits final Discord-ready text to 1,800 characters and uses `keep_alive: "0"`.

**Step 2: Run RED**
```bash
.venv/bin/python -m unittest tests/test_ollama.py -v
```

**Step 3: Implement minimal client**
- Use `urllib.request` through `asyncio.to_thread`, avoiding a new dependency.
- Define immutable `LLMResult` and `OllamaClient`.
- Explicitly set `stream: false`, deterministic temperature, `num_predict`, `keep_alive: "0"`, and `think` per route.

**Step 4: Run GREEN**
```bash
.venv/bin/python -m unittest tests/test_ollama.py -v
```

### Task 2: Build evidence-only answer composition and routing

**Objective:** Convert a reviewed support card or Prowl citations into model input while preserving deterministic safety gates.

**Files:**
- Create: `src/agent/answering.py`
- Modify: `src/agent/runtime.py`
- Test: `tests/test_answering.py`

**Step 1: Write failing tests**
- Informational and state-changing curated support answers route to Gemma.
- An explicitly diagnostic, multi-symptom question with cited Prowl evidence routes to LFM.
- Destructive and ambiguity outcomes never call an LLM.
- A model error falls back to the reviewed card answer.
- The prompt contains only the question, reviewed answer, risk label, docs URL, and selected citations; it contains no hidden reasoning instruction.

**Step 2: Run RED**
```bash
.venv/bin/python -m unittest tests/test_answering.py -v
```

**Step 3: Implement minimal composer**
- Use Gemma by default and LFM only for source-grounded diagnostics.
- Treat LFM thinking as private process metadata: never place it in embeds, exceptions, logs, or benchmark artifacts.
- Retain support-card risk badges and documentation buttons.

**Step 4: Run GREEN**
```bash
.venv/bin/python -m unittest tests/test_answering.py -v
```

### Task 3: Add channel and mention interaction policy

**Objective:** Allow normal questions in the configured support channel, preserving mention/reply-only behavior elsewhere.

**Files:**
- Modify: `bot.py`
- Modify: `src/handlers/messages.py`
- Modify: `tests/test_bot.py`
- Modify: `.env.example`

**Step 1: Write failing tests**
- A non-bot message in `SUPPORT_CHANNEL_ID` is accepted without a mention.
- The same message outside that channel is ignored unless it mentions or replies to the bot.
- Missing/invalid channel IDs fail configuration validation.
- Existing mention and reply paths remain accepted.

**Step 2: Run RED**
```bash
.venv/bin/python -m unittest tests/test_bot.py -v
```

**Step 3: Implement minimal policy**
- Extend `Config` with local Ollama values and `support_channel_id`.
- Pass a single `InteractionPolicy` object to the handler.
- Never respond to bots; retain Discord allowed-mentions protections.

**Step 4: Run GREEN**
```bash
.venv/bin/python -m unittest tests/test_bot.py -v
```

### Task 4: Repair Prowl compatibility and prevent false source answers

**Objective:** Make the bot fail closed when Prowl cannot retrieve an exact requested source rather than showing unrelated citations.

**Files:**
- Modify: `prowl.py`
- Modify: `tests/test_prowl.py`
- Modify: `tests/test_bot.py`

**Step 1: Write failing tests**
- A failed `peek` capability never falls through to unrelated source results for an explicit file request.
- Candidate-source and explicit-path results must match the requested relative path to be returned as authoritative.
- Natural-language Prowl results remain usable only when they are actual returned matches and are labeled as locations, not a direct file view.

**Step 2: Run RED**
```bash
.venv/bin/python -m unittest tests/test_prowl.py tests/test_bot.py -v
```

**Step 3: Implement minimal compatibility behavior**
- Retain existing `find` for explicit symbols.
- Treat unsupported `peek` as unavailable for exact-file display until Prowl ships an explicit bounded-read command.
- Do not fabricate source snippets from raw repository reads.

**Step 4: Run GREEN**
```bash
.venv/bin/python -m unittest tests/test_prowl.py tests/test_bot.py -v
```

### Task 5: Add model-backed integration tests and a local benchmark

**Objective:** Verify both installed models produce a user-visible final answer from the same Ryoku evidence, without revealing thinking.

**Files:**
- Create: `benchmark_llm.py`
- Create: `data/llm-evals.json`
- Test: `tests/test_llm_benchmark.py`
- Modify: `README.md`

**Step 1: Write failing validation tests**
- The benchmark schema requires a question, evidence, required answer fragments, and route.
- Final output cannot contain `<think>`, `thinking`, or hidden-reasoning fields.

**Step 2: Run RED**
```bash
.venv/bin/python -m unittest tests/test_llm_benchmark.py -v
```

**Step 3: Implement and run benchmark**
- Include a normal safe-help case for Gemma and a cited diagnostic case for LFM.
- Require evidence-derived command/path fragments.
- Serialize only final answer, latency, pass/fail, model, and route; never save hidden thinking.

**Step 4: Verify**
```bash
.venv/bin/python benchmark_llm.py --check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -B benchmark.py --mode hybrid --benchmark data/regressions.json --ryoku-repo /home/neur0map/workspace/ryoku-arch --check
```

### Task 6: Document, inspect, commit, and review

**Objective:** Leave a deployable, explained local setup without connecting to Discord.

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

**Verification:**
- Explain support-channel mode vs mention/reply mode.
- Explain that only one local model is resident at once and LFM thoughts are never surfaced.
- Document `OLLAMA_HOST`, `GEMMA_MODEL`, `LFM_MODEL`, `SUPPORT_CHANNEL_ID`, and `PROWL_AGENT_PATH`.
- Run complete tests and both model-backed benchmark routes.
- Inspect `git diff`, commit a human-style message, and push only if authenticated.
