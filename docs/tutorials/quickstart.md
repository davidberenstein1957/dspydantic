# Quickstart

Get from zero to your first extraction in 5 minutes.

## Install

```bash
pip install dspydantic
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Configure a Language Model

DSPydantic uses DSPy under the hood. Before anything else, configure the language model:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini", api_key="your-api-key")
dspy.configure(lm=lm)
```

(The API key can also come from the `OPENAI_API_KEY` environment variable.)

## Define What to Extract

Create a Pydantic model describing the data:

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")
    email: str = Field(description="Email address")
```

## Run Extraction

```python
from dspydantic import Prompter

prompter = Prompter(model=Person, model_id="openai/gpt-4o-mini")

result = prompter.run("""
    John Smith
    Age: 28
    Contact: john.smith@example.com
""")

print(result)
# Person(name='John Smith', age=28, email='john.smith@example.com')
```

**Done!** You've extracted structured data from text.

## Next Steps

- **Improve accuracy** → [Extract Structured Data](extract-structured-data.md) tutorial with optimization
- **Use text output instead** → [Extract Free-form Text](extract-free-form-text.md)
- **Use images or PDFs** → [Use Images and PDFs](../how-to/use-multimodal-inputs.md)
- **Deploy to production** → [Save and Load a Prompter](../how-to/save-and-load.md)
