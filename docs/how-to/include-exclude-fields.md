# Include or Exclude Fields

Control which fields are optimized and scored. Use `include_fields` to focus on specific fields, or `exclude_fields` to skip fields that shouldn't affect optimization.

## include_fields: Optimize Only These Fields

When you have many fields but want to optimize a subset:

```python
result = prompter.optimize(
    examples=examples,
    include_fields=["address", "total"],  # Only these (and nested) are optimized
)
```

- **Exact match**: `include_fields=["name"]` includes only `name`
- **Prefix match**: `include_fields=["address"]` includes `address`, `address.street`, `address.city`, etc.
- Reduces optimization time and API costs when you have many fields
- Non-included fields keep their original descriptions

## exclude_fields: Skip Fields in Scoring

Exclude fields when they shouldn't influence optimization but should still be extracted:

```python
result = prompter.optimize(
    examples=examples,
    exclude_fields=["metadata", "timestamp"],
)
```

Excluded fields are still extracted but don't affect the optimization score.

---

## When to Use Each

| Use Case | Parameter | Example |
|----------|-----------|---------|
| **Focus on critical fields** | `include_fields` | `["address", "total"]` |
| **Skip metadata in scoring** | `exclude_fields` | `["metadata", "timestamp"]` |
| **Combine both** | Both | Include `address`, exclude `address.internal_id` |

---

## When to Exclude Fields

| When to Exclude | Example Fields | Reason |
|-----------------|----------------|--------|
| **Metadata** | timestamps, IDs | Don't affect accuracy |
| **Non-critical** | internal notes | Reduce noise in scoring |
| **Computed** | derived values | Not extracted from input |

---

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

### 2. Choose: Include or Exclude

```python
import dspy
from dspydantic import Prompter

# Configure DSPy first
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="your-api-key"))

prompter = Prompter(model=PatientRecord)

# Option A: Exclude fields from scoring (still extracted)
result = prompter.optimize(
    examples=examples,
    exclude_fields=["metadata", "timestamp"],
)

# Option B: Only optimize specific fields
result = prompter.optimize(
    examples=examples,
    include_fields=["patient_name", "urgency", "diagnosis"],
)

# Option C: Combine both
result = prompter.optimize(
    examples=examples,
    include_fields=["patient_name", "diagnosis"],
    exclude_fields=["metadata"],
)
```

The optimization process optimizes field descriptions and prompts for the effective field set (included minus excluded).

### 3. Use Optimized Prompter

The excluded fields will still be extracted, but won't affect the optimization score:

```python
# Extract data
record = prompter.run("Patient John Doe, urgent case, diagnosed with pneumonia")
print(record.patient_name)  # Optimized
print(record.metadata)      # Still extracted, but not optimized
print(record.timestamp)     # Still extracted, but not optimized
```

---

## Impact on Optimization

| Aspect | include_fields | exclude_fields | Neither |
|--------|----------------|----------------|---------|
| **Fields Optimized** | Only specified (and nested) | All except specified | All fields |
| **Score Calculation** | Based on included only | Based on all except excluded | All fields |
| **Extraction** | All fields extracted | All fields extracted | All fields extracted |

---

## Tips

- Use `include_fields` to reduce optimization time when you have many fields
- Only exclude fields that truly don't matter for optimization
- Excluded fields are still extracted by the model
- When both are set, `exclude_fields` removes from the `include_fields` set
- Works well with [Optimize Nested Models](optimize-nested-models.md) using dot notation

---

## See Also

- [Configure Optimization Parameters](configure-optimizations.md) — Advanced optimization options
- [Optimize Nested Models](optimize-nested-models.md) — Complex structures with dot-notation field paths
- [Extract Structured Data](../tutorials/extract-structured-data.md) — Complete optimization workflow
