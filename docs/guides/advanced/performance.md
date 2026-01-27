# Performance Tuning

This guide covers performance optimization techniques for DSPydantic optimization and extraction.

## Optimization Performance

### Number of Examples

| Examples | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| **5-10** | Fast | Good | Quick iterations |
| **10-20** | Balanced | Better | Production optimization |
| **20+** | Slower | Best | Final optimization |

### Parallel Evaluation

Use multiple threads for faster optimization:

```python
result = prompter.optimize(
    examples=examples,
    num_threads=4,  # Parallel evaluation
)
```

### Optimizer Selection

DSPydantic auto-selects optimizers, but you can choose:

| Optimizer | Speed | Quality | Use Case |
|-----------|-------|---------|----------|
| `BootstrapFewShot` | Fast | Good | Small datasets (< 20 examples) |
| `BootstrapFewShotWithRandomSearch` | Medium | Better | Larger datasets (≥ 20 examples) |
| `MIPROv2` | Slow | Best | Advanced optimization needs |

```python
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch

# Faster for small datasets
optimizer = BootstrapFewShot()

# Better for larger datasets
optimizer = BootstrapFewShotWithRandomSearch(num_candidates=10)

result = prompter.optimize(
    examples=examples,
    optimizer=optimizer,
)
```

## Extraction Performance

### Caching

Cache results for repeated inputs:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_extract(text: str):
    return prompter.run(text)
```

### Batch Processing

Process multiple inputs efficiently:

```python
def batch_extract(texts: list[str]) -> list:
    return [prompter.run(text) for text in texts]
```

### Async Processing

Use async for concurrent extractions:

```python
import asyncio
from dspydantic import Prompter

async def async_extract(prompter: Prompter, text: str):
    # Note: Prompter.run() is synchronous
    # Use asyncio.to_thread for async wrapper
    return await asyncio.to_thread(prompter.run, text)

async def batch_async_extract(texts: list[str]):
    tasks = [async_extract(prompter, text) for text in texts]
    return await asyncio.gather(*tasks)
```

## Model Selection

### Faster Models

For faster extraction, use smaller models:

| Model | Speed | Accuracy | Cost | Use Case |
|-------|-------|----------|------|----------|
| **gpt-4o-mini** | Fast | Good | Low | Simple tasks |
| **gpt-4o** | Slower | Better | High | Complex tasks |

```python
# Faster but less accurate
prompter = Prompter(model=MyModel, model_id="gpt-4o-mini")

# Slower but more accurate
prompter = Prompter(model=MyModel, model_id="gpt-4o")
```

## Image Processing

### Image Size

Smaller images process faster:

```python
from PIL import Image

def resize_image(image_path: str, max_size: int = 1024):
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    return img
```

### PDF DPI

| DPI | Speed | Quality | Use Case |
|-----|-------|---------|----------|
| **150-200** | Fastest | Lower | Quick processing |
| **300** (default) | Medium | Good | Most documents |
| **600+** | Slowest | Best | Small text, detailed docs |

Lower DPI for faster PDF processing:

```python
# Faster processing
Example(pdf_path="doc.pdf", pdf_dpi=150, expected_output={...})

# Better quality
Example(pdf_path="doc.pdf", pdf_dpi=300, expected_output={...})
```

## Performance Impact

| Optimization | Before | After | Impact |
|--------------|--------|-------|--------|
| Parallel Evaluation | Sequential | 4 threads | 3-4x faster |
| Model Selection | gpt-4o | gpt-4o-mini | 2-3x faster |
| PDF DPI | 600 | 300 | 2x faster |
| Caching | None | LRU cache | Instant for repeats |

## Tips

- Use caching for repeated inputs
- Process in batches when possible
- Choose appropriate model size for your needs
- Use parallel evaluation during optimization
- Monitor API usage and costs

## See Also

- [Deploying to Production](../production/deployment.md) - Production best practices
- [Your First Optimization](../optimization/first-optimization.md) - Complete optimization workflow
- [Reference: Prompter](../../reference/api/prompter.md) - Complete API documentation
