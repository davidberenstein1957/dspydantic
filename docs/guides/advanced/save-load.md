# Production Deployment

Save optimized prompters and deploy them to production without re-running optimization.

---

## Save After Optimization

```python
from dspydantic import Prompter, Example
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    vendor: str = Field(description="Vendor name")
    total: str = Field(description="Total amount")

# Optimize
prompter = Prompter(model=Invoice, model_id="openai/gpt-4o-mini")
result = prompter.optimize(examples=[...])

# Save
prompter.save("./invoice_prompter")
```

**What gets saved:**

- Optimized field descriptions
- Optimized system and instruction prompts
- Model configuration

**What is NOT saved:**

- API keys (security)
- Examples used for optimization

---

## Load in Production

```python
import os
from dspydantic import Prompter
from myapp.models import Invoice

prompter = Prompter.load(
    "./invoice_prompter",
    model=Invoice,
    model_id="openai/gpt-4o-mini",
)

# Ready to use
invoice = prompter.run(document_text)
```

Set API key via environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

Or pass explicitly:

```python
prompter = Prompter.load(
    "./invoice_prompter",
    model=Invoice,
    model_id="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

---

## Deployment Patterns

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy saved prompter
COPY invoice_prompter/ ./invoice_prompter/

COPY app/ ./app/

CMD ["python", "-m", "app.main"]
```

```python
# app/main.py
from dspydantic import Prompter
from app.models import Invoice

prompter = Prompter.load(
    "./invoice_prompter",
    model=Invoice,
    model_id="openai/gpt-4o-mini",
)

def process(text: str) -> Invoice:
    return prompter.run(text)
```

### FastAPI Service

```python
from fastapi import FastAPI
from dspydantic import Prompter
from app.models import Invoice

app = FastAPI()

# Load once at startup
prompter = Prompter.load(
    "./invoice_prompter",
    model=Invoice,
    model_id="openai/gpt-4o-mini",
)

@app.post("/extract")
async def extract(text: str):
    result = prompter.predict_with_confidence(text)
    return {
        "data": result.data.model_dump(),
        "confidence": result.confidence
    }
```

### Serverless (AWS Lambda)

```python
from dspydantic import Prompter
from models import Invoice

# Load during cold start
prompter = None

def get_prompter():
    global prompter
    if prompter is None:
        prompter = Prompter.load(
            "/opt/invoice_prompter",  # Layer path
            model=Invoice,
            model_id="openai/gpt-4o-mini",
        )
    return prompter

def handler(event, context):
    p = get_prompter()
    result = p.run(event["text"])
    return result.model_dump()
```

---

## Versioning

Version your prompters for rollback capability:

```python
import datetime

# Save with version
version = datetime.datetime.now().strftime("%Y%m%d_%H%M")
prompter.save(f"./prompters/invoice_v{version}")

# Or semantic versioning
prompter.save("./prompters/invoice_v1.2.0")
```

Directory structure:

```
prompters/
├── invoice_v1.0.0/
├── invoice_v1.1.0/
├── invoice_v1.2.0/  # Current
└── latest -> invoice_v1.2.0/  # Symlink
```

---

## Validation Before Deploy

Test loaded prompter before deploying:

```python
def validate_prompter(prompter: Prompter, test_cases: list[dict]) -> bool:
    """Validate prompter against test cases."""
    passed = 0
    
    for case in test_cases:
        result = prompter.run(case["input"])
        
        for field, expected in case["expected"].items():
            actual = getattr(result, field)
            if actual != expected:
                print(f"FAIL: {field} - expected {expected}, got {actual}")
            else:
                passed += 1
    
    total = sum(len(c["expected"]) for c in test_cases)
    print(f"Passed {passed}/{total} checks")
    
    return passed == total

# Usage
test_cases = [
    {
        "input": "Invoice from Acme Corp. Total: $100. Due: March 1.",
        "expected": {"vendor": "Acme Corp", "total": "$100"}
    }
]

prompter = Prompter.load("./invoice_prompter", model=Invoice, model_id="openai/gpt-4o-mini")
if validate_prompter(prompter, test_cases):
    print("Ready for deployment")
```

---

## Model Upgrades

When upgrading your Pydantic model:

**Compatible changes** (safe):

- Adding optional fields with defaults
- Relaxing field types (str | None → str)

**Incompatible changes** (re-optimize):

- Adding required fields
- Changing field names
- Changing field types

```python
# Load and verify schema compatibility
try:
    prompter = Prompter.load("./invoice_prompter", model=NewInvoice, model_id="openai/gpt-4o-mini")
    # Test with sample data
    result = prompter.run(sample_text)
except Exception as e:
    print(f"Schema mismatch: {e}")
    # Re-optimize needed
```

---

## CI/CD Integration

```yaml
# .github/workflows/deploy.yml
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install dependencies
        run: pip install dspydantic pytest
      
      - name: Validate prompter
        run: pytest tests/test_prompter_validation.py
      
      - name: Deploy
        if: success()
        run: ./deploy.sh
```

```python
# tests/test_prompter_validation.py
from dspydantic import Prompter
from app.models import Invoice

def test_prompter_loads():
    prompter = Prompter.load("./invoice_prompter", model=Invoice, model_id="openai/gpt-4o-mini")
    assert prompter is not None

def test_prompter_extracts():
    prompter = Prompter.load("./invoice_prompter", model=Invoice, model_id="openai/gpt-4o-mini")
    result = prompter.run("Invoice from Test Corp. Total: $50.")
    assert result.vendor is not None
    assert result.total is not None
```

---

## See Also

- [Integration Patterns](integration-patterns.md) - FastAPI, background processing
- [Getting Started](../optimization/first-optimization.md) - Optimization workflow
- [API Reference: Prompter](../../reference/api/prompter.md) - Full documentation
