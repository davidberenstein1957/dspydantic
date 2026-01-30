# Core Concepts

DSPydantic automatically optimizes your prompts and field descriptions using your examples. This page explains the key concepts.

---

## The Core Idea

Traditional prompt engineering is trial and error. You write a prompt, test it, tweak it, repeat. DSPydantic automates this:

```mermaid
flowchart LR
    A[Your Examples] --> B[DSPy Optimizer]
    B --> C[Better Prompts]
    C --> D[Higher Accuracy]
```

You provide examples of what you want. DSPydantic finds the prompts that work best.

---

## Key Concepts

### Prompter

The `Prompter` class is your main interface. It does two things:

1. **Extract** - Get structured data from text, images, or PDFs
2. **Optimize** - Improve extraction accuracy using examples

```python
from dspydantic import Prompter

# Create a prompter
prompter = Prompter(model=Invoice, model_id="openai/gpt-4o-mini")

# Extract (works immediately)
invoice = prompter.run("Invoice #123 from Acme Corp...")

# Optimize (for better accuracy)
result = prompter.optimize(examples=examples)
```

### Examples

Examples are your training data. Each example has:

- **Input**: Text, image path, or PDF path
- **Expected output**: The correct structured data

```python
from dspydantic import Example

example = Example(
    text="Invoice from Acme Corp. Total: $1,250. Due: March 15, 2024.",
    expected_output={
        "vendor": "Acme Corp",
        "total": "$1,250",
        "due_date": "March 15, 2024"
    }
)
```

**How many examples?**

| Count | Quality | When to Use |
|-------|---------|-------------|
| 5-10 | Good | Quick prototyping |
| 10-20 | Better | Most use cases |
| 20+ | Best | Complex schemas, edge cases |

### Optimization

Optimization is the process of finding better prompts. DSPydantic:

1. Generates variations of your prompts and field descriptions
2. Tests each variation against your examples
3. Scores results using evaluators
4. Selects the best-performing combination

```python
result = prompter.optimize(examples=examples)

print(f"Before: {result.baseline_score:.0%}")    # e.g., 75%
print(f"After: {result.optimized_score:.0%}")    # e.g., 92%
```

**What gets optimized?**

| Component | Example | Impact |
|-----------|---------|--------|
| **Field descriptions** | `"Full name"` → `"Person's complete legal name as written"` | High |
| **System prompt** | `"Extract data"` → `"Extract invoice data accurately..."` | Medium |
| **Instruction prompt** | `"Get fields"` → `"Extract each field following the schema..."` | Medium |

### Evaluators

Evaluators score how well an extraction matches the expected output. Scores range from 0.0 (no match) to 1.0 (perfect match).

| Evaluator | Best For | Example |
|-----------|----------|---------|
| **Exact match** | IDs, codes, enums | `"INV-123"` |
| **Levenshtein** | Minor typos OK | `"John Doe"` vs `"Jon Doe"` |
| **Semantic similarity** | Meaning matters | `"CEO"` vs `"Chief Executive"` |
| **LLM judge** | Complex comparisons | Long text, summaries |

Default behavior is smart: enums use exact match, strings use fuzzy match.

See [Configure Evaluators](guides/evaluators/configure.md) for customization.

---

## Workflow

### Quick Start (No Optimization)

For simple cases, extract immediately:

```python
prompter = Prompter(model=Invoice, model_id="openai/gpt-4o-mini")
invoice = prompter.run(document_text)
```

### With Optimization

For better accuracy, optimize first:

```python
# 1. Define model
class Invoice(BaseModel):
    vendor: str = Field(description="Company name")
    total: str = Field(description="Amount due")

# 2. Create examples
examples = [Example(text="...", expected_output={...}), ...]

# 3. Optimize
prompter = Prompter(model=Invoice, model_id="openai/gpt-4o-mini")
result = prompter.optimize(examples=examples)

# 4. Extract with optimized prompts
invoice = prompter.run(new_document)
```

### Production Deployment

```python
# After optimization, save
prompter.save("./invoice_prompter")

# In production, load and use
prompter = Prompter.load("./invoice_prompter", model=Invoice, model_id="openai/gpt-4o-mini")
invoice = prompter.run(document)
```

---

## Input Types

| Type | Example | Use Case |
|------|---------|----------|
| **Text** | `Example(text="...")` | Documents, emails, logs |
| **Image** | `Example(image_path="photo.png")` | Screenshots, receipts |
| **PDF** | `Example(pdf_path="invoice.pdf")` | Invoices, contracts |
| **Dictionary** | `Example(text={"field": "value"})` | Template prompts |

All input types work with the same optimization process.

See [Modalities](guides/optimization/modalities.md) for details.

---

## Production Features

| Feature | Method | Use Case |
|---------|--------|----------|
| **Batch processing** | `predict_batch()` | Process many documents |
| **Async** | `apredict()`, `apredict_batch()` | High-throughput pipelines |
| **Confidence scores** | `predict_with_confidence()` | Flag uncertain extractions |
| **Caching** | `cache=True` | Reduce API costs |
| **Save/Load** | `save()`, `load()` | Deploy to production |

```python
# Batch processing
invoices = prompter.predict_batch(documents, max_workers=4)

# Async
invoice = await prompter.apredict(document)

# Confidence scores
result = prompter.predict_with_confidence(document)
if result.confidence > 0.9:
    process(result.data)
```

---

## Next Steps

- [Getting Started](guides/optimization/first-optimization.md) - Complete tutorial
- [Modalities](guides/optimization/modalities.md) - Text, images, PDFs
- [Modalities](guides/optimization/modalities.md) - Text, images, PDFs
- [Evaluators](guides/evaluators/configure.md) - Customize scoring
- [API Reference](reference/api/prompter.md) - Full documentation
