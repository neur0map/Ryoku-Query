<div align="center">

<img src="https://raw.githubusercontent.com/neur0map/ryoku-arch/main/ryoku/assets/brand/logo-mark.png" alt="Ryoku" width="160" />

# Ryoku Discord Support

**Reviewed Ryoku help inside Discord** &middot; *support answers, safety prompts, and source-backed replies.*

Ryoku Discord Support is the private Discord bot for answering reviewed Ryoku
support questions. It serves curated responses from `data/support.json`, adds
safety context for risky actions, and can search a local stable Ryoku checkout
for source-backed answers through Prowl.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-E2342A?style=for-the-badge)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-E2342A?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![discord.py](https://img.shields.io/badge/discord.py-2.6%2B-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discordpy.readthedocs.io)
[![Ryoku Docs](https://img.shields.io/badge/docs-ryoku.dev-E2342A?style=for-the-badge)](https://docs.ryoku.dev)

<kbd>[Docs](https://docs.ryoku.dev)</kbd> &middot; <kbd>[Support catalog](data/support.json)</kbd> &middot; <kbd>[Benchmarks](data/benchmark.json)</kbd> &middot; <kbd>[Discord](https://discord.gg/8KjBmUEyKA)</kbd>

</div>

---

## What it does

- Answers reviewed Ryoku support questions in Discord

## Quick start

### 1. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example env
```

### 2. Clone the stable Ryoku repo

The bot uses a local stable Ryoku checkout for source-backed answers.

```bash
git clone https://github.com/neur0map/ryoku-arch.git /path/to/ryoku
```

Point `RYOKU_REPO_PATH` in `env` at that checkout.

### 3. Initialize the Ryoku checkout for Prowl

Run this once inside the cloned Ryoku repo:

```bash
cd /path/to/ryoku
prowl-agent init --no-input --integrations none
```

### 4. Configure `env`

Set:

- `TOKEN` to your Discord bot token
- `RYOKU_REPO_PATH` to your cloned Ryoku checkout
- optionally `MODEL_NAME`, `PROWL_TIMEOUT_SECONDS`, and `PROWL_RESULT_LIMIT`

## Run

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  --env-file env \
  python -B bot.py
```

Mention the bot or reply to one of its messages when testing.

## Test

Run the unit test suite:

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B -m unittest discover -s tests -v
```

Run the regression benchmark:

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B benchmark.py --mode hybrid \
  --benchmark data/regressions.json \
  --ryoku-repo /path/to/ryoku \
  --check
```

## Benchmark

Run the full benchmark with replay sampling:

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B benchmark.py --mode hybrid \
  --ryoku-repo /path/to/ryoku \
  --replays 3 --check
```
