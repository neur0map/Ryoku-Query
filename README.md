# Ryoku Discord Support

Private Discord support bot for Ryoku.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example env
```

Set the token and stable Ryoku checkout in `env`.

Initialize the Ryoku checkout once:

```bash
cd /path/to/ryoku
prowl-agent init --no-input --integrations none
```

## Run

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  --env-file env \
  python -B bot.py
```

Mention the bot or reply to one of its messages when testing.

## Test

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B -m unittest discover -s tests -v
```

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B benchmark.py --mode hybrid \
  --benchmark data/regressions.json \
  --ryoku-repo /path/to/ryoku \
  --check
```

## Benchmark

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B benchmark.py --mode hybrid \
  --ryoku-repo /path/to/ryoku \
  --replays 3 --check
```
