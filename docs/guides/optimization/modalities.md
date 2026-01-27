# Optimization Modalities

All modalities (text, images, PDFs) work with **structured output and a Pydantic schema**, or you can optimize **without a Pydantic schema** for string output. Modalities define how you format inputs in your examples; output structure is decided by whether you use a schema or not.

Use this page as the **input format reference**. Then choose how you want to optimize:

- **[With Pydantic Schema](with-pydantic-schema.md)** — Structured output with your Pydantic model
- **[Without Pydantic Schema](without-pydantic-schema.md)** — String output; single `"output"` field
- **[Prompt Templates](prompt-templates.md)** — Dynamic prompts with `{placeholders}` from dictionary input

## Input Formats

Each modality uses different fields on `Example`. Use the format that matches your input type.

| Modality | Example field(s) | Output format |
|----------|------------------|----------------|
| **Text** | `text` (str) | Dict with schema fields, or `expected_output` (str) when no schema |
| **Images** | `image_path` or `image_base64` | Dict with schema fields, or str when no schema |
| **PDFs** | `pdf_path`, optional `pdf_dpi` | Dict with schema fields, or str when no schema |
| **Templates** | `text` (dict) | Dict with schema fields; keys become `{placeholders}` in prompts |

## Text

Use `text` with a string for documents, emails, or any plain text.

```python
from dspydantic import Example

# With Pydantic schema — expected_output is a dict matching your model
Example(
    text="Goldman Sachs processed a $2.5M equity trade for Tesla Inc. on March 15, 2024.",
    expected_output={
        "broker": "Goldman Sachs",
        "amount": "$2.5M",
        "security": "Tesla Inc.",
        "date": "March 15, 2024"
    }
)

# Without Pydantic schema — expected_output is a string
Example(text="The movie was excellent.", expected_output="positive")
```

## Images

Use `image_path` (file path) or `image_base64` (base64 string).

```python
from dspydantic import Example

# With image path
Example(
    image_path="digit_5.png",
    expected_output={"digit": 5}  # or expected_output="5" when no schema
)

# With base64
import base64
def load_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

Example(
    image_base64=load_b64("digit_5.png"),
    expected_output={"digit": 5}
)
```

## PDFs

Use `pdf_path`. Optionally set `pdf_dpi` (default 300). PDFs are converted to images for processing.

```python
from dspydantic import Example

Example(
    pdf_path="invoice_001.pdf",
    pdf_dpi=300,  # optional
    expected_output={
        "invoice_number": "INV-2024-001",
        "date": "2024-01-15",
        "total_amount": "$1,250.00"
    }
)

# With text context
Example(
    text="Q1 2024 invoice",
    pdf_path="invoice.pdf",
    expected_output={...}
)
```

## Dictionary text (for prompt templates)

Use `text` with a **dict** when you use [prompt templates](prompt-templates.md). Keys are used as `{placeholders}` in prompts.

```python
Example(
    text={
        "review": "Amazing camera!",
        "product": "iPhone 15",
        "category": "smartphone"
    },
    expected_output={"sentiment": "positive", "rating": 4}
)
```

## Next steps

- **[With Pydantic Schema](with-pydantic-schema.md)** — Optimize using a Pydantic model and the input formats above
- **[Without Pydantic Schema](without-pydantic-schema.md)** — Optimize for string output with `model=None`
- **[Prompt Templates](prompt-templates.md)** — Optimize prompts with `{placeholders}` and dictionary input
- [Your First Optimization](first-optimization.md) — End-to-end workflow
- [Reference: Prompter](../../reference/api/prompter.md) — API details
