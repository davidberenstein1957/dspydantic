# Optimize Without a Pydantic Schema

Optimize prompts and field descriptions when you want to optimize output without a schema. DSPydantic creates a blank schema with a single field called "output" and optimizes that.

## When to Use Without a Pydantic Schema

Use `model=None` when:

- You want to optimize output without a schema
- Your examples have string outputs (not structured dictionaries)
- You simply want to optimize prompts for string extraction
- You don't need structured fields - just a single "output" field

## Problem

You have examples with string outputs and want to optimize extraction without defining a schema upfront.

## Solution

Pass `model=None` to `Prompter`. DSPydantic will create a blank schema with a single field called "output" (type: str) and optimize that. This works when your examples have string `expected_output` values.

## Steps

### 1. Create Examples with String Outputs

```python
from dspydantic import Example

examples = [
    Example(
        text="The movie was excellent with great acting.",
        expected_output="positive"
    ),
    Example(
        text="Terrible plot and boring characters.",
        expected_output="negative"
    ),
    Example(
        text="It was okay, nothing special.",
        expected_output="neutral"
    ),
]
```

Notice: Examples must have string `expected_output` values. No Pydantic schema definition needed - DSPydantic creates a blank schema with a single "output" field.

### 2. Configure DSPy and Create Prompter

```python
import dspy
from dspydantic import Prompter

# Configure DSPy with your language model
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

# Pass model=None to auto-generate the model
prompter = Prompter(model=None)

result = prompter.optimize(examples=examples)
```

DSPydantic automatically:
1. Creates a blank schema with a single field called "output" (type: str)
2. Optimizes the field description and prompts for string extraction

### 3. Extract Data (Outcome)

After optimization, extract string output efficiently:

```python
# Extract from new text
data = prompter.extract("The film was amazing!")
print(data.output)  # "positive" (or whatever the optimized output is)
```

## How It Works

| Step | What Happens |
|------|--------------|
| **1. Create Blank Schema** | DSPydantic creates a simple OutputModel with a single field called "output" (type: str) |
| **2. Optimize** | Optimizes the "output" field description and prompts for string extraction |

## What Gets Optimized

| Parameter | What Gets Optimized | Impact |
|-----------|-------------------|--------|
| "output" Field Description | Description for the single output field | High - direct extraction accuracy |
| System Prompt | Overall context | Medium - task understanding |
| Instruction Prompt | Task instructions | Medium - extraction guidance |

## Important Notes

- **Only works with string outputs**: Your examples must have string `expected_output` values
- **Single field only**: Creates a blank schema with one field called "output"
- **Not schema inference**: This does not discover or infer schemas from examples - it simply optimizes string output without a schema
- After optimization, you can inspect the generated model: `prompter.model`
- For structured data extraction, define a proper Pydantic model instead
