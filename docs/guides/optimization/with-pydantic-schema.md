# Optimize With a Pydantic Schema

Optimize for **structured output** by passing a Pydantic model to `Prompter`. DSPydantic optimizes field descriptions and prompts to maximize extraction accuracy. Use any input modality — format examples as in [Optimization Modalities](modalities.md).

## When to use

- You want structured dict/object output
- Examples have `expected_output` as a **dict** matching your model
- You need typed, validated extraction

## Workflow

### 1. Define your model

Create a Pydantic model with field descriptions:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Review(BaseModel):
    """Extract review data."""
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    rating: int = Field(description="Rating from 1 to 5")
    summary: str = Field(description="Brief summary of the review")
```

### 2. Create examples

Use dict `expected_output` matching your model's fields:

```python
from dspydantic import Example

examples = [
    Example(
        text="Amazing product! Works perfectly and exceeded expectations.",
        expected_output={
            "sentiment": "positive",
            "rating": 5,
            "summary": "Product exceeded expectations"
        }
    ),
    Example(
        text="Broke after two weeks. Complete waste of money.",
        expected_output={
            "sentiment": "negative",
            "rating": 1,
            "summary": "Product broke quickly"
        }
    ),
    Example(
        text="It's okay. Does what it says but nothing special.",
        expected_output={
            "sentiment": "neutral",
            "rating": 3,
            "summary": "Average product, meets basic expectations"
        }
    ),
]
```

### 3. Optimize

```python
import dspy
from dspydantic import Prompter

dspy.configure(lm=dspy.LM("openai/gpt-4o", api_key="your-api-key"))

prompter = Prompter(model=Review)
result = prompter.optimize(examples=examples)
```

### 4. Run

```python
data = prompter.run("This is the best purchase I've ever made!")
print(data)
# Review(sentiment='positive', rating=5, summary='Best purchase ever')
```

## Images and PDFs

Same pattern with [image](modalities.md#images) or [PDF](modalities.md#pdfs) inputs: use dict `expected_output` matching your model.

```python
from pydantic import BaseModel, Field

class Digit(BaseModel):
    digit: int = Field(description="The handwritten digit (0-9)")

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice ID")
    total: str = Field(description="Total amount")

# Images
examples = [
    Example(image_path="digit_5.png", expected_output={"digit": 5}),
    Example(image_path="digit_3.png", expected_output={"digit": 3}),
]

prompter = Prompter(model=Digit)
result = prompter.optimize(examples=examples)
digit = prompter.run(image_path="new_digit.png")  # Digit(digit=7)

# PDFs
examples = [
    Example(pdf_path="invoice.pdf", expected_output={"invoice_number": "INV-001", "total": "$500"}),
]

prompter = Prompter(model=Invoice)
result = prompter.optimize(examples=examples)
inv = prompter.run(pdf_path="new_invoice.pdf")  # Invoice(...)
```

## How it works

| Step | What happens |
|------|----------------|
| `model=YourModel` | Schema fields are used for structured extraction |
| Optimize | Field descriptions and prompts are optimized for accuracy |
| Run | `prompter.run(...)` returns an instance of your model |

## What gets optimized

| What | Impact |
|------|--------|
| Field descriptions | High |
| System / instruction prompts | Medium |

## Tips

- Every example must have `expected_output` as a dict matching your model
- Field descriptions guide the LLM — be specific
- Use `Literal` for categorical fields with known values
- Use `| None` for optional fields
- For string output without a schema, use [Without Pydantic Schema](without-pydantic-schema.md)
- [Reference: Example](../../reference/api/types.md#example)

## See also

- [Optimization Modalities](modalities.md) — Input formats for text, images, PDFs
- [Without Pydantic Schema](without-pydantic-schema.md) — String output with `model=None`
- [Prompt Templates](prompt-templates.md) — Dynamic prompts with `{placeholders}`
- [Nested Models](../advanced/nested-models.md) — Complex schemas with nested Pydantic models
