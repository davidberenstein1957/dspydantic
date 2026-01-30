# Getting Started

Get from zero to optimized extraction in 5 minutes.

---

## Prerequisites

```bash
pip install dspydantic
```

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

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

- Field descriptions guide the LLM—be specific
- Use `Literal` for categorical fields
- Use `| None` for optional fields
- Lists work for multi-value fields

---

## Step 2: Create Examples

Provide examples of input text and expected output:

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

```python
from dspydantic import Prompter

prompter = Prompter(
    model=JobPosting,
    model_id="openai/gpt-4o-mini",
)

result = prompter.optimize(examples=examples)
```

Optimization takes 1-5 minutes depending on example count.

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

---

## Step 5: Extract

Use your optimized prompter:

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
| Different input types | [Modalities](modalities.md) |
| Images and PDFs | [Modalities - Images/PDFs](modalities.md#images) |
| Customize evaluation | [Configure Evaluators](../evaluators/configure.md) |
| Complex schemas | [Nested Models](../advanced/nested-models.md) |
| Production deployment | [Save and Load](../advanced/save-load.md) |
| Integration patterns | [Integration Patterns](../advanced/integration-patterns.md) |

---

## Troubleshooting

**Low accuracy after optimization?**

- Add more diverse examples
- Check that examples are correct
- Try a more capable model (`gpt-4o` vs `gpt-4o-mini`)

**Optimization takes too long?**

- Reduce example count for initial testing
- Use `gpt-4o-mini` for faster iterations

**API key issues?**

```python
# Set key explicitly
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="sk-..."))
```

See [Configure Models](../advanced/configure-models.md) for more options.
