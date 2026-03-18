# Input Formats Reference

Complete reference for all input modalities supported by DSPydantic.

## Format Comparison

| Modality | Field | Type | Example |
|----------|-------|------|---------|
| **Text** | `text` | str | `"sample text"` |
| **Image (file)** | `image_path` | str | `"image.png"` |
| **Image (base64)** | `image_base64` | str | `"data:image/png;base64,..."` |
| **PDF** | `pdf_path` | str | `"document.pdf"` |
| **PDF + DPI** | `pdf_path`, `pdf_dpi` | str, int | `pdf_dpi=300` |
| **Dict/Template** | `text` | dict | `{"key": "value"}` |

---

## Text

```python
from dspydantic import Example

Example(text="sample text here", expected_output={...})
```

**Best for:** Documents, emails, messages, reviews

**Tips:**
- Longest text field is `text` (up to model limit)
- Supports any plain text format
- Use with structured output (dict) or freeform (model=None)

---

## Images

### File Path

```python
Example(
    image_path="path/to/image.png",
    expected_output={...}
)
```

**Supported formats:** `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`

### Base64 Encoded

```python
import base64

with open("image.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

Example(
    image_base64=b64,
    expected_output={...}
)
```

**Best for:** Handwriting, diagrams, screenshots, visual inspection

**Tips:**
- Images should be clear and well-lit
- File paths work for local files
- Base64 useful for cloud/API scenarios

---

## PDFs

### File Path

```python
Example(
    pdf_path="path/to/document.pdf",
    expected_output={...}
)
```

### With DPI Control

```python
Example(
    pdf_path="document.pdf",
    pdf_dpi=300,  # Higher = better OCR, slower
    expected_output={...}
)
```

**DPI Guide:**
- `150`: Fast, lower quality
- `300`: Balanced (default)
- `600`: Best quality, slower

**Best for:** Forms, invoices, documents, reports, contracts

**Tips:**
- Works best with text-based PDFs
- Use `pdf_dpi=300` for scanned documents
- Multi-page PDFs supported

---

## Dictionaries (Templates)

```python
Example(
    text={
        "category": "smartphone",
        "product": "iPhone 15",
        "review": "Great camera!"
    },
    expected_output={...}
)
```

**Best for:** Dynamic prompts with `{placeholders}`

**Usage in prompts:**
```python
prompter.optimize(
    examples=examples,
    system_prompt="Analyze {category} reviews",
    instruction_prompt="Review: {review}"
)
```

**Tips:**
- Keys become placeholders in prompts
- Placeholders are case-sensitive
- All examples must have same keys

---

## Combining Inputs

You can combine text with other modalities:

```python
# Image + text context
Example(
    image_path="invoice.png",
    text="Date: 2024-01-15",
    expected_output={...}
)

# PDF + text context
Example(
    pdf_path="form.pdf",
    text="Customer: John Smith",
    expected_output={...}
)
```

---

## Choosing Input Format

| Task | Format | Reason |
|------|--------|--------|
| Documents | Text | Most reliable |
| Scanned docs | PDF | Built-in OCR |
| Handwriting | Image | Visual |
| Structured text | Dict | Dynamic prompts |
| Forms | PDF or Image | Layout matters |

---

## API Reference

- **Text:** `Example(text=str, ...)`
- **Image:** `Example(image_path=str, ...)` or `Example(image_base64=str, ...)`
- **PDF:** `Example(pdf_path=str, pdf_dpi=int, ...)`
- **Dict:** `Example(text=dict, ...)`

---

## See Also

- [Extract Structured Data](../tutorials/extract-structured-data.md) — Full workflow
- [Use Images and PDFs](../how-to/use-multimodal-inputs.md) — How-to guide
