# Extract Structured Data

Learn the complete workflow: define a schema, optimize with examples, and extract with high accuracy.

## When to Use

- You want **structured output** (a typed object, not just text)
- Your data has multiple fields with specific types
- You need **validated extraction** (Pydantic ensures correct types)

If you just want text output, see [Extract Free-form Text](extract-free-form-text.md) instead.

---

## Step 1: Define Your Model

Create a Pydantic model describing what you want to extract:

```python
from pydantic import BaseModel, Field
from typing import Literal

class JobPosting(BaseModel):
    """Extract structured data from job postings."""
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="Job location")
    salary_range: str | None = Field(description="Salary range if mentioned")
    experience_years: str | None = Field(description="Required years of experience")
    employment_type: Literal["full_time", "part_time", "contract", "internship"] = Field(
        description="Type of employment"
    )
    remote: bool = Field(description="Whether remote work is available")
    skills: list[str] = Field(description="Required skills or technologies")
```

**Tips:**

- Field descriptions guide the LLM — be specific
- Use `Literal` for categorical fields with known values
- Use `| None` for optional fields
- Use lists for multi-value fields

---

## Step 2: Create Examples

Provide examples of input text and expected output as **dicts**:

```python
from dspydantic import Example

examples = [
    Example(
        text="""
        Senior Software Engineer at TechCorp

        Location: San Francisco, CA (Hybrid - 3 days onsite)
        Salary: $180,000 - $220,000

        We're looking for an experienced engineer with 5+ years of experience
        in Python and cloud infrastructure. Strong background in AWS, Kubernetes,
        and CI/CD pipelines required.

        Full-time position with competitive benefits.
        """,
        expected_output={
            "title": "Senior Software Engineer",
            "company": "TechCorp",
            "location": "San Francisco, CA",
            "salary_range": "$180,000 - $220,000",
            "experience_years": "5+ years",
            "employment_type": "full_time",
            "remote": True,
            "skills": ["Python", "AWS", "Kubernetes", "CI/CD"]
        }
    ),
    Example(
        text="""
        Data Analyst Intern - FinanceHub

        NYC Office, No Remote

        3-month internship for current students. Must know SQL and Excel.
        Experience with Tableau is a plus.
        """,
        expected_output={
            "title": "Data Analyst Intern",
            "company": "FinanceHub",
            "location": "NYC Office",
            "salary_range": None,
            "experience_years": None,
            "employment_type": "internship",
            "remote": False,
            "skills": ["SQL", "Excel", "Tableau"]
        }
    ),
    Example(
        text="""
        Contract DevOps Engineer

        RemoteFirst Inc. | 100% Remote | $85-95/hr

        6-month contract. Looking for someone with 3 years experience in
        Terraform, Docker, and GitHub Actions. Azure certification preferred.
        """,
        expected_output={
            "title": "Contract DevOps Engineer",
            "company": "RemoteFirst Inc.",
            "location": "100% Remote",
            "salary_range": "$85-95/hr",
            "experience_years": "3 years",
            "employment_type": "contract",
            "remote": True,
            "skills": ["Terraform", "Docker", "GitHub Actions", "Azure"]
        }
    ),
]
```

**How many examples?**

- **5-10**: Good for simple models
- **10-20**: Recommended for most cases
- **20+**: For complex schemas or edge cases

---

## Step 3: Optimize

Create a prompter and optimize with your examples:

```python
import dspy
from dspydantic import Prompter

# Configure language model (see Configure a Language Model tutorial)
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="your-api-key"))

# Create prompter
prompter = Prompter(
    model=JobPosting,
    model_id="openai/gpt-4o-mini",
)

# Optimize with examples
result = prompter.optimize(examples=examples)
```

**What gets optimized:**

| What | Impact |
|------|--------|
| Field descriptions | High — clarifies what each field should extract |
| System/instruction prompts | Medium — guides overall extraction behavior |

**How the default mode works:**

By default (`sequential=False`), all fields and prompts are optimized together in a single pass for speed. For higher quality, use `sequential=True` to optimize each field independently (starting with deepest-nested fields), then optimize system and instruction prompts.

See [Configure Optimization Parameters](../how-to/configure-optimizations.md) for more options like `early_stopping_patience`, `auto_generate_prompts`, and `compile_kwargs`.

Optimization takes 1-5 minutes depending on example count and model.

---

## Step 4: Check Results

```python
print(f"Before: {result.baseline_score:.0%}")
print(f"After:  {result.optimized_score:.0%}")
print(f"API calls: {result.api_calls}")
print(f"Tokens: {result.total_tokens:,}")
```

**Typical output:**

```
Before: 72%
After:  91%
API calls: 47
Tokens: 28,450
```

View optimized descriptions:

```python
for field, desc in result.optimized_descriptions.items():
    print(f"{field}: {desc}")
```

You'll see how the optimizer refined the field descriptions to guide the model better.

---

## Step 5: Extract

Use your optimized prompter on new data:

```python
job = prompter.run("""
    ML Engineer - AI Startup

    Boston, MA or Remote
    $150K-200K base + equity

    Join our team building next-gen recommendation systems.
    Need 4+ years with PyTorch, transformers, and production ML.
    Full-time. Start immediately.
""")

print(job)
# JobPosting(
#     title='ML Engineer',
#     company='AI Startup',
#     location='Boston, MA or Remote',
#     salary_range='$150K-200K base + equity',
#     experience_years='4+ years',
#     employment_type='full_time',
#     remote=True,
#     skills=['PyTorch', 'transformers', 'production ML']
# )
```

The result is a fully typed `JobPosting` instance. Pydantic validates all fields before returning.

---

## Step 6: Save for Production

```python
# Save the optimized prompter
prompter.save("./job_parser")

# Later, in production:
prompter = Prompter.load(
    "./job_parser",
    model=JobPosting,
    model_id="openai/gpt-4o-mini"
)

job = prompter.run(new_posting_text)
```

See [Save and Load a Prompter](../how-to/save-and-load.md) for detailed deployment instructions.

---

## Going Further: Images and PDFs

The same workflow works with images and PDFs. Just use different input modalities:

### Images

```python
from pydantic import BaseModel, Field

class Digit(BaseModel):
    digit: int = Field(description="The handwritten digit (0-9)")

examples = [
    Example(image_path="digit_5.png", expected_output={"digit": 5}),
    Example(image_path="digit_3.png", expected_output={"digit": 3}),
]

prompter = Prompter(model=Digit)
result = prompter.optimize(examples=examples)
digit = prompter.run(image_path="new_digit.png")
print(digit.digit)  # 7
```

### PDFs

```python
class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice ID")
    total: str = Field(description="Total amount")

examples = [
    Example(pdf_path="invoice_001.pdf", expected_output={"invoice_number": "INV-001", "total": "$500"}),
    Example(pdf_path="invoice_002.pdf", expected_output={"invoice_number": "INV-002", "total": "$750"}),
]

prompter = Prompter(model=Invoice)
result = prompter.optimize(examples=examples)
inv = prompter.run(pdf_path="new_invoice.pdf")
print(inv.invoice_number)  # "INV-003"
```

Full input format details are in [Use Images and PDFs](../how-to/use-multimodal-inputs.md).

---

## Quick Reference

| Method | Purpose |
|--------|---------|
| `Prompter(model, model_id)` | Create prompter |
| `prompter.optimize(examples)` | Optimize with examples |
| `prompter.run(text)` | Extract from text |
| `prompter.predict_batch(texts)` | Batch extraction |
| `prompter.save(path)` | Save optimized state |
| `Prompter.load(path, model, model_id)` | Load saved prompter |

---

## Next Steps

| Topic | Guide |
|-------|-------|
| Extract text instead | [Extract Free-form Text](extract-free-form-text.md) |
| Dynamic prompts | [Optimize with Prompt Templates](use-prompt-templates.md) |
| Nested models | [Optimize Nested Models](../how-to/optimize-nested-models.md) |
| Customize evaluation | [Configure Evaluators](../how-to/configure-evaluators.md) |
| Production deployment | [Save and Load a Prompter](../how-to/save-and-load.md) |
| Integration patterns | [Integrate with Applications](../how-to/integrate-with-applications.md) |

---

## Troubleshooting

**Low accuracy after optimization?**

- Add more diverse examples (aim for 10-20)
- Check that your examples are correct
- Try a more capable model (`gpt-4o` vs `gpt-4o-mini`)
- Review the optimized field descriptions to see if they make sense

**Optimization takes too long?**

- Reduce example count for initial testing
- Use `gpt-4o-mini` for faster iterations
- Use single-pass mode (default) or limit trials with `compile_kwargs={"num_trials": 5}`

**API key issues?**

```python
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="sk-..."))
```

See [Configure a Language Model](configure-language-models.md) for all options.
