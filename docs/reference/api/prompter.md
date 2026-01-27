# Prompter

Unified class for optimizing and predicting with Pydantic models.

::: dspydantic.prompter.Prompter
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - optimize
        - predict
        - save
        - load
        - from_optimization_result

## Overview

The `Prompter` class combines optimization and prediction functionality in a single interface. Use it to optimize field descriptions and prompts, then predict structured data from text, images, or PDFs.

## Basic Usage

```python
import dspy
from dspydantic import Prompter, Example
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(description="User name")
    age: int = Field(description="User age")

# Configure DSPy first
lm = dspy.LM("openai/gpt-4o", api_key="your-key")
dspy.configure(lm=lm)

# Create prompter
prompter = Prompter(model=User)

# Optimize
result = prompter.optimize(
    examples=[Example(text="John Doe, 30", expected_output={"name": "John Doe", "age": 30})]
)

# Save
prompter.save("./my_prompter")

# Load (DSPy must be configured before loading)
prompter = Prompter.load("./my_prompter", model=User)

# Predict from text
data = prompter.predict("Jane Smith, 25")

# Predict from image
data = prompter.predict(image_path="photo.png")

# Predict from PDF
data = prompter.predict(pdf_path="document.pdf")

# Predict with prompt templates
data = prompter.predict(text={"review": "Great product!", "category": "electronics"})
```

## See Also

- [Save and Load Prompters](../../guides/advanced/save-load.md)
- [Optimization Modalities](../../guides/optimization/modalities.md)
- [Optimize with Templates](../../guides/optimization/templates.md)
- [Your First Optimization](../../guides/optimization/first-optimization.md)
