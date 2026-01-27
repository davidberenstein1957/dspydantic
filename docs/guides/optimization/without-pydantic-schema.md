# Optimize Without a Pydantic Schema

Optimize for **string output** by passing `model=None`. DSPydantic uses a single field `"output"` (str) and optimizes prompts and that description. Use any input modality — format examples as in [Optimization Modalities](modalities.md).

## When to use

- You want string output, not structured dicts
- Examples have `expected_output` as a **string**
- You don’t need a Pydantic model

## Workflow

### 1. Create examples

Use string `expected_output` and inputs in one of the [input formats](modalities.md#input-formats) (text, images, PDFs).

```python
from dspydantic import Example

examples = [
    Example(text="The movie was excellent with great acting.", expected_output="positive"),
    Example(text="Terrible plot and boring characters.", expected_output="negative"),
    Example(text="It was okay, nothing special.", expected_output="neutral"),
]
```

### 2. Optimize

```python
import dspy
from dspydantic import Prompter

dspy.configure(lm=dspy.LM("openai/gpt-4o", api_key="your-api-key"))

prompter = Prompter(model=None)
result = prompter.optimize(examples=examples)
```

### 3. Extract

```python
data = prompter.extract("The film was amazing!")
print(data.output)  # e.g. "positive"
```

## Images and PDFs

Same pattern with [image](modalities.md#images) or [PDF](modalities.md#pdfs) inputs: use string `expected_output`.

```python
# Images
examples = [
    Example(image_path="digit_5.png", expected_output="5"),
    Example(image_path="digit_3.png", expected_output="3"),
]

# PDFs
examples = [
    Example(pdf_path="invoice.pdf", expected_output="INV-2024-001"),
]

prompter = Prompter(model=None)
result = prompter.optimize(examples=examples)

digit = prompter.extract(image_path="new_digit.png")   # digit.output
inv   = prompter.extract(pdf_path="new_invoice.pdf")  # inv.output
```

## How it works

| Step | What happens |
|------|----------------|
| `model=None` | A minimal schema with one field `"output"` (str) is used |
| Optimize | That field’s description and the prompts are optimized for string extraction |
| Extract | `prompter.extract(...)` returns an object whose `.output` is the string |

## What gets optimized

| What | Impact |
|------|--------|
| `"output"` field description | High |
| System / instruction prompts | Medium |

## Tips

- Every example must have `expected_output` as a str
- No custom schema — only the built-in `"output"` field
- For structured extraction, use a Pydantic model and [With Pydantic Schema](with-pydantic-schema.md)
- [Reference: Example](../../reference/api/types.md#example)

## See also

- [Optimization Modalities](modalities.md) — Input formats for text, images, PDFs
- [With Pydantic Schema](with-pydantic-schema.md) — Structured output with a Pydantic model
- [Prompt Templates](prompt-templates.md) — Dynamic prompts with `{placeholders}`
