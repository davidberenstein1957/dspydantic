# Configure a Language Model

Learn how to set up language models for DSPydantic. This is the foundation — you must configure DSPy before you can use any other feature.

## The Concept

DSPydantic uses DSPy, a library for optimizing language model prompts. Before you create a prompter or run optimization, you must tell DSPy which language model to use:

```python
import dspy

# 1. Create a language model object
lm = dspy.LM("openai/gpt-4o-mini", api_key="your-api-key")

# 2. Configure DSPy to use it
dspy.configure(lm=lm)

# 3. Now prompters will use this model
from dspydantic import Prompter
prompter = Prompter(model=MyModel)  # Uses gpt-4o-mini
```

The configuration is **global** — once you call `dspy.configure()`, all prompters use that model until you configure a different one.

## Step 1: Use OpenAI (the default)

OpenAI is the most common choice. It's reliable and offers good performance.

```python
import dspy
import os

# Option A: Use environment variable (recommended)
# Set OPENAI_API_KEY in your shell or .env file
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Option B: Pass the key directly
lm = dspy.LM("openai/gpt-4o-mini", api_key="sk-...")
dspy.configure(lm=lm)
```

**Available OpenAI models:**
- `openai/gpt-4o` — Best quality (slower, higher cost)
- `openai/gpt-4o-mini` — Good quality, faster, lower cost (good for starting out)
- `openai/gpt-4-turbo` — Balanced quality and speed
- `openai/gpt-3.5-turbo` — Fastest, lower quality

**Which to use?**
- For development and prototyping → `gpt-4o-mini`
- For production with high quality → `gpt-4o`

## Step 2: Switch to Anthropic (just one line)

Notice how easy it is to switch? This shows the beauty of the configuration model — the rest of your code stays the same.

```python
import dspy
import os

# Option A: Use environment variable
os.environ["ANTHROPIC_API_KEY"] = "your-key"
lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929")

# Option B: Pass directly
lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929", api_key="your-key")

dspy.configure(lm=lm)
```

**Available Anthropic models:**
- `anthropic/claude-sonnet-4-5-20250929` — Latest Claude Sonnet (excellent reasoning)
- `anthropic/claude-opus-3` — High quality
- `anthropic/claude-haiku-3` — Fast, cost-effective

**Why switch?**
- Anthropic models often reason better on complex extraction tasks
- Anthropic has competitive pricing
- Good for comparison testing

## Step 3: Use a Local Model (Ollama)

Want to run everything locally? Ollama makes it easy.

### Setup

1. **Install Ollama** from https://ollama.ai
2. **Pull a model** (in your terminal):
   ```bash
   ollama pull llama3.2
   ```
3. **Start the Ollama server** (keep it running):
   ```bash
   ollama serve
   ```

### Configure

```python
import dspy

lm = dspy.LM(
    "ollama_chat/llama3.2",
    api_base="http://localhost:11434"
)
dspy.configure(lm=lm)
```

Now your prompter uses the local model. No API calls, no costs, fully private.

### Why use local models?

- **Privacy** — data stays on your machine
- **Cost** — free after initial setup
- **Offline** — works without internet
- **Latency** — can be faster depending on your hardware

**Trade-off:** Local models are generally less capable than cloud models for complex extraction tasks.

## Bonus: Other Cloud Providers

DSPy supports other providers too. Here's the pattern:

### Google Gemini

```python
import dspy

lm = dspy.LM("gemini/gemini-2.5-pro-preview-03-25", api_key="your-key")
dspy.configure(lm=lm)
```

### Databricks

```python
import dspy

lm = dspy.LM(
    "databricks/model-name",
    api_key="your-key",
    api_base="https://your-workspace.cloud.databricks.com"
)
dspy.configure(lm=lm)
```

## What You've Learned

1. **DSPy configuration is global** — one call to `dspy.configure()` applies to all prompters
2. **Model config is not saved** — when you load a saved prompter, you must reconfigure DSPy before using it
3. **Switching models is one line** — the rest of your code stays the same
4. **You have options** — cloud (OpenAI, Anthropic, Gemini) or local (Ollama)
5. **Environment variables are best** — store API keys outside your code

## API Key Best Practices

**Never hardcode API keys:**

```python
# ❌ Bad
lm = dspy.LM("openai/gpt-4o-mini", api_key="sk-abc123xyz")  # Don't commit this!

# ✅ Good
import os
api_key = os.getenv("OPENAI_API_KEY")
lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
```

**Use .env files during development:**

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
```

```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env
lm = dspy.LM("openai/gpt-4o-mini")  # Uses env var
```

## Speed vs Quality Trade-off

When optimizing vs extracting, you might use different models:

```python
import dspy

# During optimization: use a strong model
lm_optimize = dspy.LM("openai/gpt-4o", api_key="your-key")
dspy.configure(lm=lm_optimize)

prompter = Prompter(model=MyModel)
result = prompter.optimize(examples=examples)  # Uses gpt-4o

# During extraction: switch to a faster model
lm_extract = dspy.LM("openai/gpt-4o-mini", api_key="your-key")
dspy.configure(lm=lm_extract)

data = prompter.run("text")  # Uses gpt-4o-mini
```

This balances quality (strong model for learning) with cost (fast model for inference).

## Next Steps

- **First extraction** — [Quickstart](quickstart.md) or [Extract Structured Data](extract-structured-data.md)
- **Extract text** — [Extract Free-form Text](extract-free-form-text.md)
- **Optimize with evaluators** — [Configure Evaluators](../how-to/configure-evaluators.md)
- **Deep dive on optimizers** — [Reference: Optimizers](../reference/optimizers.md)
