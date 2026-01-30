# DSPydantic

**Stop manually tuning prompts. Let your data optimize them.**

DSPydantic automatically optimizes your Pydantic model prompts and field descriptions using DSPy. Extract structured data from text, images, and PDFs with higher accuracy and less effort.

[![PyPI](https://img.shields.io/pypi/v/dspydantic)](https://pypi.org/project/dspydantic/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-online-green.svg)](https://davidberenstein1957.github.io/dspydantic/)

## The Problem

You've defined a Pydantic model. You're using an LLM to extract data. But:

- Your prompts are guesswork—trial and error until something works
- Accuracy varies wildly depending on input phrasing
- Every new use case means more manual prompt engineering

## The Solution

DSPydantic takes your examples and **automatically finds the best prompts** for your use case:

```python
from pydantic import BaseModel, Field
from dspydantic import Prompter, Example

class Invoice(BaseModel):
    vendor: str = Field(description="Company that issued the invoice")
    total: str = Field(description="Total amount due")
    due_date: str = Field(description="Payment due date")

prompter = Prompter(model=Invoice, model_id="openai/gpt-4o-mini")

# Optimize with examples
result = prompter.optimize(examples=[
    Example(
        text="Invoice from Acme Corp. Total: $1,250.00. Due: March 15, 2024.",
        expected_output={"vendor": "Acme Corp", "total": "$1,250.00", "due_date": "March 15, 2024"}
    ),
])

# Extract with optimized prompts
invoice = prompter.run("Consolidated Energy Partners | Invoice Total $3,200 | Due 2024-05-30")
```

**Typical improvement: 10-30% higher accuracy** with the same LLM.

## Installation

```bash
pip install dspydantic
```

## Quick Start

### Extract Data (No Optimization)

For simple cases, extract immediately:

```python
from pydantic import BaseModel, Field
from dspydantic import Prompter

class Contact(BaseModel):
    name: str = Field(description="Person's full name")
    email: str = Field(description="Email address")

prompter = Prompter(model=Contact, model_id="openai/gpt-4o-mini")

contact = prompter.run("Reach out to Sarah Chen at sarah.chen@techcorp.io")
# Contact(name='Sarah Chen', email='sarah.chen@techcorp.io')
```

### Optimize for Better Accuracy

When accuracy matters, optimize with examples:

```python
from dspydantic import Example

examples = [
    Example(text="...", expected_output={...}),
    # 5-20 examples typically enough
]

result = prompter.optimize(examples=examples)
print(f"Accuracy: {result.baseline_score:.0%} → {result.optimized_score:.0%}")
```

### Deploy to Production

```python
# Save optimized prompter
prompter.save("./invoice_prompter")

# Load in production
prompter = Prompter.load("./invoice_prompter", model=Invoice, model_id="openai/gpt-4o-mini")
invoice = prompter.run(new_document)
```

## Why DSPydantic?

| Feature | DSPydantic | Manual Prompting |
|---------|------------|------------------|
| **Automatic optimization** | ✅ Data-driven | ❌ Trial and error |
| **Pydantic native** | ✅ Full type safety | ⚠️ JSON only |
| **Multi-modal** | ✅ Text, images, PDFs | ⚠️ Text only |
| **Production ready** | ✅ Save/load, batch, async | ❌ Manual |
| **Confidence scores** | ✅ Per-extraction | ❌ No |

**Built on:** [DSPy](https://dspy.ai/) (Stanford's optimization framework) + [Pydantic](https://docs.pydantic.dev/) (Python data validation)

## Input Types

```python
# Text
Example(text="Invoice from Acme...", expected_output={...})

# Images
Example(image_path="receipt.png", expected_output={...})

# PDFs
Example(pdf_path="contract.pdf", expected_output={...})
```

## Production Features

```python
# Caching (reduce API costs)
prompter = Prompter(model=Invoice, model_id="openai/gpt-4o-mini", cache=True)

# Batch processing
invoices = prompter.predict_batch(documents, max_workers=4)

# Async
invoice = await prompter.apredict(document)

# Confidence scores
result = prompter.predict_with_confidence(document)
if result.confidence > 0.9:
    process(result.data)
```

## Documentation

Full documentation at [davidberenstein1957.github.io/dspydantic](https://davidberenstein1957.github.io/dspydantic/)

- [Getting Started](https://davidberenstein1957.github.io/dspydantic/guides/optimization/first-optimization/) - First extraction in 5 minutes
- [Use Cases](https://davidberenstein1957.github.io/dspydantic/use-cases/) - Real-world examples
- [Cookbook](https://davidberenstein1957.github.io/dspydantic/cookbook/) - Copy-paste patterns
- [API Reference](https://davidberenstein1957.github.io/dspydantic/reference/api/prompter/) - Full documentation

## License

Apache 2.0

## Contributing

Contributions welcome! [Open an issue](https://github.com/davidberenstein1957/dspydantic/issues) or submit a pull request.
