# Use Images and PDFs

Extract from images and PDF documents using DSPydantic.

## Images

### Image Paths

```python
from dspydantic import Example, Prompter

class Digit(BaseModel):
    digit: int = Field(description="Handwritten digit 0-9")

examples = [
    Example(image_path="digit_5.png", expected_output={"digit": 5}),
    Example(image_path="digit_3.png", expected_output={"digit": 3}),
]

prompter = Prompter(model=Digit)
result = prompter.optimize(examples=examples)

digit = prompter.run(image_path="new_digit.png")
print(digit.digit)  # 7
```

### Base64 Images

```python
import base64

with open("image.png", "rb") as f:
    b64_image = base64.b64encode(f.read()).decode()

example = Example(
    image_base64=b64_image,
    expected_output={"digit": 5}
)
```

---

## PDFs

### PDF Paths

```python
class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice number")
    total: str = Field(description="Total amount due")

examples = [
    Example(pdf_path="invoice_1.pdf", expected_output={"invoice_number": "INV-001", "total": "$500"}),
    Example(pdf_path="invoice_2.pdf", expected_output={"invoice_number": "INV-002", "total": "$750"}),
]

prompter = Prompter(model=Invoice)
result = prompter.optimize(examples=examples)

invoice = prompter.run(pdf_path="new_invoice.pdf")
print(invoice.invoice_number)  # "INV-003"
```

### PDF with DPI

Control PDF rendering quality:

```python
example = Example(
    pdf_path="document.pdf",
    pdf_dpi=300,  # Higher = better quality, slower
    expected_output={...}
)
```

---

## PDFs with Text Context

Combine text context with PDF:

```python
example = Example(
    pdf_path="form.pdf",
    text="Customer: John Smith, Date: 2024-01-15",  # Additional context
    expected_output={...}
)
```

---

## Tips

- Images should be clear and well-lit
- PDFs work best with text-based content (not scanned images)
- Use `pdf_dpi=300` for better OCR quality
- Combine text context with PDFs for better accuracy
- Test with a few examples before optimizing with many

---

## See Also

- [Extract Structured Data](../tutorials/extract-structured-data.md) — Full optimization workflow
- [Configure Evaluators](configure-evaluators.md) — Customize evaluation
