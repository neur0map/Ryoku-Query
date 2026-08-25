# Ryoku Dev : Discord similarity bot

A minimal Discord bot that:

1. Reads live message content using Discord's **Message Content Intent**.
2. Embeds each incoming message with `minishlab/potion-code-16M-v2`.
3. Compares it against the examples in `faq.json` using cosine similarity.
4. Sends a Discord embed when the best score is greater than or equal to `THRESHOLD`.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

```bash
.venv\Scripts\activate # Windows
source .venv/bin/activate # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and update the values.

### 5. Ensure the Message Content Intent is enabled:
Application → Bot → Privileged Gateway Intents → Message Content Intent → ON

### 6. Run the bot

```bash
python bot.py
```
