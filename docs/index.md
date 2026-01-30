# DSPydantic

**Stop manually tuning prompts. Let your data optimize them.**

DSPydantic automatically optimizes your Pydantic model prompts and field descriptions using DSPy. Extract structured data from text, images, and PDFs with higher accuracy and less effort.

---

## The Problem

You've defined a Pydantic model. You're using an LLM to extract data. But:

- Your prompts are guesswork—trial and error until something works
- Accuracy varies wildly depending on input phrasing
- Every new use case means more manual prompt engineering
- You can't measure what "good enough" actually means

## The Solution

DSPydantic takes your examples and **automatically finds the best prompts** for your use case.

```python
from pydantic import BaseModel, Field
from dspydantic import Prompter, Example

class SupportTicket(BaseModel):
    category: str = Field(description="Issue category")
    priority: str = Field(description="Urgency level")
    summary: str = Field(description="Brief summary")

prompter = Prompter(model=SupportTicket, model_id="openai/gpt-4o-mini")

result = prompter.optimize(examples=[
    Example(
        text="I've been trying to log in for 2 HOURS and keep getting 'invalid password'!!!",
        expected_output={"category": "account", "priority": "high", "summary": "Login failures despite correct password"}
    ),
    Example(
        text="Quick question - was charged $49.99 but I'm on the $29.99 plan. Not urgent.",
        expected_output={"category": "billing", "priority": "low", "summary": "Unexpected charge discrepancy"}
    ),
])

print(f"Accuracy: {result.baseline_score:.0%} → {result.optimized_score:.0%}")
# Accuracy: 68% → 94%
```

---

## See It Work: Before vs After

The magic is in what DSPydantic discovers. Here's a real optimization result:

| Field | Your Description | Optimized Description |
|-------|------------------|----------------------|
| `priority` | `"Urgency level"` | `"Urgency based on user frustration signals, deadline mentions, and business impact"` |
| `category` | `"Issue category"` | `"Primary issue type: billing, technical, account, or product"` |
| `summary` | `"Brief summary"` | `"One-sentence summary focusing on the core problem, not symptoms"` |

The optimizer found that **specificity matters**: vague descriptions like "urgency level" led to inconsistent results, while explicit guidance about "frustration signals" and "deadline mentions" dramatically improved accuracy.

---

## Quick Start

### Install

```bash
pip install dspydantic
```

### Extract Without Optimization

For simple cases, extract immediately:

```python
from pydantic import BaseModel, Field
from dspydantic import Prompter

class Contact(BaseModel):
    name: str = Field(description="Person's full name")
    email: str = Field(description="Email address")
    company: str = Field(description="Company or organization")

prompter = Prompter(model=Contact, model_id="openai/gpt-4o-mini")

contact = prompter.run("Reach out to Sarah Chen at sarah.chen@techcorp.io, she's the CTO at TechCorp.")
# Contact(name='Sarah Chen', email='sarah.chen@techcorp.io', company='TechCorp')
```

### Optimize for Better Accuracy

When accuracy matters, provide examples:

```python
from dspydantic import Example

examples = [
    Example(text="...", expected_output={...}),
    Example(text="...", expected_output={...}),
]

result = prompter.optimize(examples=examples)
print(f"Accuracy improved: {result.baseline_score:.0%} → {result.optimized_score:.0%}")
```

### Deploy to Production

```python
prompter.save("./my_prompter")

prompter = Prompter.load("./my_prompter", model=Contact, model_id="openai/gpt-4o-mini")
contact = prompter.run(new_document)
```

---

---

## Multi-Modal: Text, Images, PDFs

### From Images

```python
Example(
    image_path="receipt.png",
    expected_output={"merchant": "Starbucks", "total": "$5.75", "date": "2024-03-15"}
)
```

### From PDFs

```python
Example(
    pdf_path="contract.pdf",
    expected_output={"parties": ["Acme Inc", "ClientCo"], "effective_date": "2024-01-01"}
)
```

---

## Production Features

```python
prompter = Prompter(model=Invoice, model_id="openai/gpt-4o-mini", cache=True)

documents = ["doc1...", "doc2...", "doc3..."]
invoices = prompter.predict_batch(documents, max_workers=4)

invoice = await prompter.apredict(document)

result = prompter.predict_with_confidence(document)
if result.confidence > 0.9:
    process(result.data)
else:
    flag_for_review(result.data)
```

---

## Why DSPydantic?

| Feature | DSPydantic | Manual Prompting | Instructor |
|---------|------------|------------------|------------|
| **Automatic optimization** | ✅ Data-driven | ❌ Trial and error | ❌ Manual |
| **Works without labels** | ✅ Judge-based | ❌ No | ❌ No |
| **Multi-modal** | ✅ Text, images, PDFs | ⚠️ Manual setup | ⚠️ Text focused |
| **Measurable accuracy** | ✅ Before/after scores | ❌ No metrics | ❌ No metrics |

### Built on Proven Foundations

- **[DSPy](https://dspy.ai/)** - Stanford's framework for optimizing LLM programs
- **[Pydantic](https://docs.pydantic.dev/)** - The standard for Python data validation

---

## Get Started

| Guide | Description |
|-------|-------------|
| [**Getting Started**](guides/optimization/first-optimization.md) | First extraction in 5 minutes |
| [**Core Concepts**](core-concepts.md) | Understand optimization and evaluation |
| [**Modalities**](guides/optimization/modalities.md) | Text, images, and PDFs |
| [**Production**](guides/advanced/save-load.md) | Save, load, batch, and async |

---

## Guides

### Optimization

- [Your First Optimization](guides/optimization/first-optimization.md) - Complete workflow
- [Optimization Modalities](guides/optimization/modalities.md) - Text, images, PDFs
- [Prompt Templates](guides/optimization/prompt-templates.md) - Dynamic prompts with placeholders
- [Without Pydantic Schema](guides/optimization/without-pydantic-schema.md) - String output

### Evaluators

- [Configure Evaluators](guides/evaluators/configure.md) - Per-field evaluation
- [Evaluator Selection](guides/evaluators/selection.md) - Choose the right evaluator
- [Custom Evaluators](guides/evaluators/custom.md) - Build your own

### Advanced

- [Nested Models](guides/advanced/nested-models.md) - Complex schemas
- [Field Exclusion](guides/advanced/field-exclusion.md) - Skip fields in evaluation
- [Save and Load](guides/advanced/save-load.md) - Production deployment

---

## Concepts

- [How Optimization Works](concepts/optimization.md) - Deep dive
- [Understanding Evaluators](concepts/evaluators.md) - Evaluation strategies
- [Architecture](concepts/architecture.md) - System design

---

## API Reference

- [Prompter](reference/api/prompter.md) - Main interface
- [Types](reference/api/types.md) - Example, OptimizationResult
- [Extractor](reference/api/extractor.md) - Field extraction
- [Evaluators](reference/api/evaluators.md) - Evaluation system

---

## Installation

```bash
pip install dspydantic
```

**Requirements:** Python 3.10+

---

## License

Apache 2.0

## Contributing

Contributions welcome! [Open an issue](https://github.com/davidberenstein1957/dspydantic/issues) or submit a pull request.
