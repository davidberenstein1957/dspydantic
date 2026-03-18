# Deploy to Production

Deploy optimized prompters in Docker, FastAPI, and serverless environments.

## FastAPI Service

```python
from fastapi import FastAPI
from pydantic import BaseModel
from dspydantic import Prompter
import dspy

app = FastAPI()

# Load once at startup
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
prompter = Prompter.load(
    "./my_prompter",
    model=MyModel,
    model_id="openai/gpt-4o-mini"
)

class ExtractionRequest(BaseModel):
    text: str

class ExtractionResponse(BaseModel):
    data: dict
    confidence: float

@app.post("/extract")
async def extract(request: ExtractionRequest):
    result = prompter.run(request.text)
    return ExtractionResponse(
        data=result.dict(),
        confidence=1.0
    )
```

---

## Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy prompter
COPY my_prompter ./my_prompter

# Copy app
COPY app.py .

# Set API key from environment
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t my-prompter-api .

# Run
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 my-prompter-api
```

---

## AWS Lambda

```python
import json
import dspy
from dspydantic import Prompter

# Load at cold start (global scope)
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
prompter = Prompter.load(
    "./my_prompter",
    model=MyModel,
    model_id="openai/gpt-4o-mini"
)

def lambda_handler(event, context):
    text = json.loads(event['body'])['text']
    result = prompter.run(text)

    return {
        'statusCode': 200,
        'body': json.dumps(result.dict())
    }
```

---

## Validation Before Deploy

```python
def validate_prompter():
    """Test prompter before deploying."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    prompter = Prompter.load("./my_prompter", model=MyModel)

    # Test cases
    test_cases = [
        "sample input 1",
        "sample input 2",
    ]

    for test_input in test_cases:
        try:
            result = prompter.run(test_input)
            assert result is not None
            print(f"✓ {test_input}")
        except Exception as e:
            print(f"✗ {test_input}: {e}")
            return False

    return True

if __name__ == "__main__":
    assert validate_prompter(), "Validation failed"
```

---

## CI/CD Integration

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - run: pip install -r requirements.txt

      - name: Validate prompter
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python validate.py

      - name: Run tests
        run: pytest tests/

      - name: Build and push Docker image
        run: docker build -t my-prompter-api . && docker push my-prompter-api
```

---

## Tips

- Always validate before deploying
- Use environment variables for API keys
- Load prompters once (not per request) for performance
- Monitor API usage in production
- Have a rollback plan for new versions

---

## See Also

- [Save and Load a Prompter](save-and-load.md) — How to persist prompters
- [Integrate with Applications](integrate-with-applications.md) — Real-world patterns
