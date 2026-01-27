# Field Exclusion

This guide shows you how to exclude certain fields from affecting the optimization score. Excluded fields are still extracted but don't influence optimization.

## When to Exclude Fields

| When to Exclude | Example Fields | Reason |
|-----------------|----------------|--------|
| **Metadata** | timestamps, IDs | Don't affect accuracy |
| **Non-critical** | internal notes | Reduce noise in scoring |
| **Computed** | derived values | Not extracted from input |

Exclude fields when they shouldn't influence optimization but should still be extracted.

## Problem

You have fields like metadata or timestamps that shouldn't affect optimization scoring, but you still want them extracted.

## Solution

Use `exclude_fields` parameter to exclude fields from evaluation while still extracting them. This allows optimization to focus on fields that matter.

## Steps

### 1. Define Your Model

```python
from pydantic import BaseModel, Field
from typing import Literal

class PatientRecord(BaseModel):
    patient_name: str = Field(description="Patient full name")
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="Urgency level of the case"
    )
    diagnosis: str = Field(description="Primary diagnosis")
    metadata: str = Field(description="Internal metadata")  # Not important for evaluation
    timestamp: str = Field(description="Record timestamp")  # Not important for evaluation
```

### 2. Exclude Fields from Evaluation

```python
from dspydantic import Prompter

# Configure DSPy first
import dspy
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

prompter = Prompter(model=PatientRecord)

result = prompter.optimize(
    examples=examples,
    exclude_fields=["metadata", "timestamp"],  # These won't affect scoring
)
```

The optimization process will optimize field descriptions **and prompts** for all fields except the excluded ones.

### 3. Use Optimized Prompter

The excluded fields will still be extracted, but won't affect the optimization score:

```python
# Extract data
record = prompter.run("Patient John Doe, urgent case, diagnosed with pneumonia")
print(record.patient_name)  # Optimized
print(record.metadata)     # Still extracted, but not optimized
```

## Impact on Optimization

| Aspect | With Exclusion | Without Exclusion |
|--------|---------------|-------------------|
| **Fields Optimized** | Only included fields | All fields |
| **Optimization Focus** | Critical fields only | All fields equally |
| **Score Calculation** | Based on included fields | Based on all fields |
| **Extraction** | All fields extracted | All fields extracted |

## Tips

- Only exclude fields that truly don't matter for optimization
- Excluded fields are still extracted by the model
- Use this sparingly - most fields should be optimized
- See [Reference: Prompter](../../reference/api/prompter.md) for details

## See Also

- [Nested Models](nested-models.md) - Optimize complex structures
- [Your First Optimization](../optimization/first-optimization.md) - Complete optimization workflow
- [Reference: Prompter](../../reference/api/prompter.md) - Complete API documentation
