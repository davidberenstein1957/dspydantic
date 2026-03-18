# Extract Free-form Text

Learn how to extract unstructured text output without defining a Pydantic model.

## When You'd Use This

Not every extraction task needs a rigid schema. Sometimes you want:

- **Sentiment analysis** — just a sentiment label or score, not a structured object
- **Text summarization** — a free-form summary of a document
- **Open-ended Q&A** — answers that vary in structure
- **Classification** — assigning one of several text categories

Instead of a Pydantic model with typed fields, you optimize a single text output using `model=None`.

## Step 1: Create Examples

Prepare examples with **string** `expected_output`:

```python
from dspydantic import Example

examples = [
    Example(
        text="The movie was absolutely brilliant. Amazing performances and a gripping story.",
        expected_output="positive"
    ),
    Example(
        text="Terrible acting, boring plot, waste of time.",
        expected_output="negative"
    ),
    Example(
        text="It was okay. Some good parts, some slow scenes.",
        expected_output="neutral"
    ),
    Example(
        text="A masterpiece! Every scene was perfect. Highly recommend it.",
        expected_output="positive"
    ),
    Example(
        text="Not what I expected. Didn't really enjoy it.",
        expected_output="negative"
    ),
]
```

**Tips:**
- Every example must have `expected_output` as a **string**, not a dict
- Use multiple examples (5-20 recommended) to teach the model your desired output format
- Examples should be representative of real data you'll process

## Step 2: Configure and Optimize

```python
import dspy
from dspydantic import Prompter

# Configure language model (see Configure a Language Model tutorial)
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="your-api-key"))

# Create prompter with model=None
prompter = Prompter(model=None)

# Optimize with your examples
result = prompter.optimize(examples=examples)

print(f"Baseline accuracy: {result.baseline_score:.0%}")
print(f"Optimized accuracy: {result.optimized_score:.0%}")
```

**Output example:**
```
Baseline accuracy: 60%
Optimized accuracy: 92%
```

The optimizer improved the field description and system/instruction prompts to better guide the model toward your desired output format.

## Step 3: Extract

Use your optimized prompter to extract from new text:

```python
data = prompter.run("""
    I loved this film. The director did an amazing job,
    and the cinematography was stunning.
""")

print(data.output)  # "positive"
```

Access the result with `.output` — that's the string you optimized for.

## With Images and PDFs

The same pattern works with images and PDFs. Just use the appropriate input format:

### Images

```python
examples = [
    Example(image_path="digit_5.png", expected_output="5"),
    Example(image_path="digit_3.png", expected_output="3"),
    Example(image_path="digit_7.png", expected_output="7"),
]

prompter = Prompter(model=None)
result = prompter.optimize(examples=examples)

digit = prompter.run(image_path="new_digit.png")
print(digit.output)  # e.g., "5"
```

### PDFs

```python
examples = [
    Example(pdf_path="invoice_001.pdf", expected_output="INV-2024-001"),
    Example(pdf_path="invoice_002.pdf", expected_output="INV-2024-002"),
]

prompter = Prompter(model=None)
result = prompter.optimize(examples=examples)

invoice = prompter.run(pdf_path="new_invoice.pdf")
print(invoice.output)  # e.g., "INV-2024-003"
```

## How It Works

Under the hood, DSPydantic creates a minimal internal schema with a single field `"output"` (a string field). When you optimize, it:

1. Optimizes the field description for `"output"` to clarify what you want
2. Optimizes the system and instruction prompts
3. Tests against your examples to measure accuracy

When you call `prompter.run(...)`, the model generates text and DSPydantic extracts the string result.

## What Gets Optimized

| What | Impact |
|------|--------|
| `"output"` field description | High — describes the desired output format |
| System/instruction prompts | Medium — guides the overall extraction behavior |

## Tips

- Use consistent output formats across examples (e.g., always "positive", "negative", "neutral" for sentiment)
- When output should be numeric or structured, consider using a Pydantic model instead (see [Extract Structured Data](extract-structured-data.md))
- For complex multi-step reasoning, use [Prompt Templates](use-prompt-templates.md) for dynamic prompts
- To save your optimized prompter for production, see [Save and Load a Prompter](../how-to/save-and-load.md)

## Next Steps

| Topic | Guide |
|-------|-------|
| Structured data with types | [Extract Structured Data](extract-structured-data.md) |
| Dynamic prompts with placeholders | [Optimize with Prompt Templates](use-prompt-templates.md) |
| Images and PDFs in detail | [Use Images and PDFs](../how-to/use-multimodal-inputs.md) |
| Production deployment | [Save and Load a Prompter](../how-to/save-and-load.md) |
| Customize evaluation | [Configure Evaluators](../how-to/configure-evaluators.md) |
