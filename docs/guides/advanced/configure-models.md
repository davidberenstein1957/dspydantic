# Configure Models

This guide covers how to configure DSPy language models for use with DSPydantic. You must configure DSPy with `dspy.configure(lm=lm)` before creating prompters or running optimization.

## DSPy Model Configuration

DSPydantic uses DSPy's language model configuration. You must configure DSPy **before** creating prompters:

```python
import dspy

# Configure DSPy with your language model
# Pass temperature and max_tokens for generation behavior
lm = dspy.LM(
    "openai/gpt-4o",
    api_key="your-api-key",
    temperature=1.0,
    max_tokens=16000,
)
dspy.configure(lm=lm)

# Now create prompters - they will use the configured model
from dspydantic import Prompter
prompter = Prompter(model=MyModel)
```

**Important**: Model configuration is **not saved** with prompters. You must configure DSPy before loading saved prompters.

## Cloud Providers

### OpenAI

Configure OpenAI models:

```python
import dspy
import os

# Set API key via environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Or pass directly (with temperature and max_tokens)
lm = dspy.LM(
    "openai/gpt-4o",
    api_key="your-api-key",
    temperature=1.0,
    max_tokens=16000,
)
dspy.configure(lm=lm)
```

**Available Models**:
- `openai/gpt-4o` - Best quality, slower
- `openai/gpt-4o-mini` - Faster, good quality
- `openai/gpt-4-turbo` - Balanced option
- `openai/gpt-3.5-turbo` - Fastest, lower quality

### Anthropic

Configure Anthropic Claude models:

```python
import dspy
import os

# Set API key via environment variable
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Or pass directly
lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929", api_key="your-api-key")
dspy.configure(lm=lm)
```

**Available Models**:
- `anthropic/claude-sonnet-4-5-20250929` - Latest Claude Sonnet
- `anthropic/claude-opus-3` - High quality
- `anthropic/claude-haiku-3` - Fast, cost-effective

### Google Gemini

Configure Google Gemini models:

```python
import dspy
import os

# Set API key via environment variable
os.environ["GEMINI_API_KEY"] = "your-api-key"

# Or pass directly
lm = dspy.LM("gemini/gemini-2.5-pro-preview-03-25", api_key="your-api-key")
dspy.configure(lm=lm)
```

**Available Models**:
- `gemini/gemini-2.5-pro-preview-03-25` - Latest Gemini Pro
- `gemini/gemini-1.5-pro` - Stable version

### Databricks

Configure Databricks models:

```python
import dspy
import os

# Set API key and base URL
os.environ["DATABRICKS_API_KEY"] = "your-api-key"
os.environ["DATABRICKS_API_BASE"] = "https://your-workspace.cloud.databricks.com"

# Or pass directly
lm = dspy.LM("databricks/model-name", api_key="your-api-key", api_base="https://...")
dspy.configure(lm=lm)
```

## Local Models

### Ollama

Run models locally with Ollama:

```python
import dspy

# 1. Install Ollama: https://ollama.ai
# 2. Run model: ollama run llama3.2

# Configure DSPy to use Ollama
lm = dspy.LM(
    "ollama_chat/llama3.2",
    api_base="http://localhost:11434",
    api_key=""  # Not needed for local
)
dspy.configure(lm=lm)
```

**Setup Steps**:
1. Install Ollama from https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. Run the model: `ollama run llama3.2`
4. Configure DSPy as shown above

### SGLang

Use SGLang for fast local inference:

```python
import dspy

# 1. Install SGLang and launch server with your model
# 2. Server runs on http://localhost:7501/v1

# Configure DSPy to use SGLang (OpenAI-compatible endpoint)
lm = dspy.LM(
    "openai/model-name",  # Use any model name
    api_base="http://localhost:7501/v1",
    model_type="chat"
)
dspy.configure(lm=lm)
```

**Setup Steps**:
1. Install SGLang: `pip install sglang`
2. Launch server with your model
3. Configure DSPy to use the OpenAI-compatible endpoint

### vLLM

Use vLLM for high-performance local inference:

```python
import dspy

# 1. Install vLLM and launch server
# 2. Server runs on http://localhost:8000/v1

# Configure DSPy to use vLLM (OpenAI-compatible endpoint)
lm = dspy.LM(
    "openai/model-name",  # Use any model name
    api_base="http://localhost:8000/v1"
)
dspy.configure(lm=lm)
```

**Setup Steps**:
1. Install vLLM: `pip install vllm`
2. Launch server with your model
3. Configure DSPy to use the OpenAI-compatible endpoint

## Model Selection Guide

### Speed vs Accuracy

| Model Type | Speed | Accuracy | Cost | Use Case |
|------------|-------|----------|------|----------|
| **gpt-4o-mini** | Fast | Good | Low | Simple tasks, quick iterations |
| **gpt-4o** | Slower | Better | High | Complex tasks, production |
| **claude-haiku** | Fast | Good | Low | Fast extraction |
| **claude-sonnet** | Medium | Best | High | High-quality extraction |
| **Local (Ollama)** | Varies | Good | Free | Privacy-sensitive, offline |

### Provider Comparison

| Provider | Speed | Quality | Cost | Best For |
|----------|-------|---------|------|----------|
| **OpenAI** | Fast | High | Medium | General use, production |
| **Anthropic** | Medium | Very High | High | Complex reasoning |
| **Gemini** | Fast | High | Low | Cost-effective |
| **Local** | Varies | Good | Free | Privacy, offline |

## API Key Management

### Environment Variables (Recommended)

Store API keys securely:

```python
import os
import dspy

# Set in environment (never commit to git)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)
```

### Direct Configuration

For testing only:

```python
import dspy

lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)
```

**Security Best Practices**:
- Never commit API keys to version control
- Use environment variables or secrets management
- Rotate keys regularly
- Use different keys for development and production

## Model Selection for Optimization

### During Optimization

Use a capable model for optimization:

```python
import dspy

# Use a strong model for optimization
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

prompter = Prompter(model=MyModel)
result = prompter.optimize(examples=examples)  # Uses gpt-4o
```

### During Extraction

You can use a faster model for extraction:

```python
import dspy

# Switch to faster model for extraction
lm_fast = dspy.LM("openai/gpt-4o-mini", api_key="your-api-key")
dspy.configure(lm=lm_fast)

# Load optimized prompter
prompter = Prompter.load("./saved_prompter")
data = prompter.run("text")  # Uses gpt-4o-mini
```

## Performance Impact

| Configuration | Speed | Accuracy | Cost |
|---------------|-------|----------|------|
| **gpt-4o** | Baseline | Best | High |
| **gpt-4o-mini** | 2-3x faster | Good | Low |
| **Local (Ollama)** | Varies | Good | Free |

## Tips

- Configure DSPy **before** creating prompters
- Use environment variables for API keys
- Choose model based on speed vs accuracy needs
- Use stronger models for optimization, faster models for extraction
- Test with different models to find the best balance
- Monitor API usage and costs

## See Also

- [Configure Optimizations](configure-optimizations.md) - Optimization parameters
- [Save and Load](save-load.md) - Saving optimized prompters
- [Your First Optimization](../optimization/first-optimization.md) - Complete workflow
- [DSPy Language Models Documentation](https://dspy.ai/learn/programming/language_models/) - Complete DSPy model guide
