# Key Concepts

Learn the fundamental concepts of DSPydantic.

## What is DSPydantic?

DSPydantic = **DSPy** (DSPy language model optimization) + **Pydantic** (data validation).

It automatically optimizes your prompts using examples, improving extraction accuracy without manual prompt engineering.

---

## Core Components

### Prompter

The main class for extraction and optimization.

```python
from dspydantic import Prompter

prompter = Prompter(
    model=MyPydanticModel,
    model_id="openai/gpt-4o-mini"
)

# Optimize with examples
result = prompter.optimize(examples=examples)

# Extract from new data
extracted = prompter.run("text")
```

**Key methods:**
- `optimize()` — Learn from examples
- `run()` — Extract from new data
- `predict_batch()` — Batch extraction
- `save()` / `load()` — Persist for production

---

### Example

Input-output pairs for optimization.

```python
from dspydantic import Example

Example(
    text="input text",
    expected_output={...}  # dict or Pydantic model
)
```

Examples teach the optimizer what good extraction looks like.

---

### OptimizationResult

Returned by `prompter.optimize()`.

```python
result = prompter.optimize(examples=examples)

result.baseline_score          # Score before optimization
result.optimized_score         # Score after optimization
result.api_calls              # API calls made
result.total_tokens           # Tokens used
result.optimized_descriptions # Optimized field descriptions
```

---

## The Optimization Loop

DSPydantic optimizes three things:

1. **Field Descriptions** — Makes them clearer for the LLM
2. **System Prompt** — Sets overall context
3. **Instruction Prompt** — Guides extraction step-by-step

**How it works:**

```
Define Model
    ↓
Create Examples
    ↓
Optimize (test variations, measure accuracy)
    ↓
Optimized Prompter (ready to use)
    ↓
Extract from New Data
```

---

## Evaluators

Measure extraction quality during optimization.

| Evaluator | Best For |
|-----------|----------|
| `exact` | IDs, exact matches |
| `levenshtein` | Names (allow variations) |
| `text_similarity` | Descriptions |
| `score_judge` | Complex evaluation |
| Custom | Domain-specific logic |

The optimizer tests field descriptions and prompts against examples, using evaluators to measure quality.

---

## Input Types

DSPydantic supports multiple input modalities:

| Type | Example |
|------|---------|
| Text | `text="sample text"` |
| Image | `image_path="image.png"` or `image_base64=...` |
| PDF | `pdf_path="document.pdf"` |
| Dict | `text={"key": "value"}` for templates |

---

## Output Types

**With Pydantic model:**
```python
prompter = Prompter(model=MyModel)
result = prompter.run(text)  # Returns MyModel instance
```

**Without model (text output):**
```python
prompter = Prompter(model=None)
result = prompter.run(text)  # Returns result.output (str)
```

---

## Production Features

| Feature | Purpose |
|---------|---------|
| Save/Load | Persist optimized prompters |
| Batch | Extract from multiple inputs |
| Async | Non-blocking extraction |
| Confidence | Measure extraction certainty |
| Caching | Reduce API calls |

---

## Quick Example

```python
from pydantic import BaseModel, Field
from dspydantic import Example, Prompter
import dspy

# 1. Define model
class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")

# 2. Create examples
examples = [
    Example(text="John is 30 years old", expected_output={"name": "John", "age": 30}),
    Example(text="Sarah is 25", expected_output={"name": "Sarah", "age": 25}),
]

# 3. Configure and optimize
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
prompter = Prompter(model=Person)
result = prompter.optimize(examples=examples)

print(f"Accuracy: {result.optimized_score:.0%}")

# 4. Extract from new data
person = prompter.run("Alice is 28 years old")
print(person)  # Person(name='Alice', age=28)
```

---

## Next Steps

- **[How Optimization Works](how-optimization-works.md)** — Deep dive into optimization
- **[Understanding Evaluators](understanding-evaluators.md)** — Evaluation mechanics
- **[Choosing an Evaluator](choosing-an-evaluator.md)** — Evaluator selection guide
