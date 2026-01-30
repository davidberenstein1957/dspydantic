# Prompter

Unified class for optimizing and extracting with Pydantic models.

::: dspydantic.prompter.Prompter
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - optimize
        - predict
        - run
        - predict_with_confidence
        - predict_batch
        - apredict
        - apredict_batch
        - save
        - load
        - from_optimization_result

## Overview

The `Prompter` class combines optimization and extraction functionality in a single interface. Use it to optimize field descriptions and prompts, then extract structured data from text, images, or PDFs.

## Basic Usage

```python
from dspydantic import Prompter, Example
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(description="User name")
    age: int = Field(description="User age")

# Simple setup with model_id (auto-configures DSPy)
prompter = Prompter(model=User, model_id="openai/gpt-4o-mini")

# Extract directly (no optimization required)
data = prompter.run("Jane Smith, 25")

# Or with optimization for better accuracy
result = prompter.optimize(
    examples=[Example(text="John Doe, 30", expected_output={"name": "John Doe", "age": 30})]
)
data = prompter.run("Jane Smith, 25")

# Save and load
prompter.save("./my_prompter")
prompter = Prompter.load("./my_prompter", model=User, model_id="openai/gpt-4o-mini")
```

## Production Features

```python
# Enable caching to reduce API costs
prompter = Prompter(model=User, model_id="openai/gpt-4o-mini", cache=True)

# Batch extraction (parallel)
texts = ["Alice, 25", "Bob, 30", "Carol, 35"]
users = prompter.predict_batch(texts, max_workers=4)

# Async extraction
user = await prompter.apredict("John Doe, 30")

# Extraction with confidence score
result = prompter.predict_with_confidence("John Doe, 30")
print(f"Confidence: {result.confidence:.0%}")

# Multi-modal extraction
data = prompter.run(image_path="photo.png")
data = prompter.run(pdf_path="document.pdf")
```

## See Also

- [Save and Load Prompters](../../guides/advanced/save-load.md)
- [Optimization Modalities](../../guides/optimization/modalities.md)
- [Optimize with Templates](../../guides/optimization/prompt-templates.md)
- [Your First Optimization](../../guides/optimization/first-optimization.md)
