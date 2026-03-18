# Optimize Nested Models

Handle Pydantic models with nested structures. DSPydantic automatically optimizes all levels of nesting.

## When to Use

| Use Case | Example | Benefit |
|----------|---------|---------|
| **Addresses** | `customer.address.street` | Hierarchical data organization |
| **Complex Data** | `company.location.address` | Multi-level accuracy |
| **Related Entities** | `order.items.product` | Structured relationships |

## Problem

You have nested Pydantic models and want to optimize field descriptions for all levels.

## Solution

DSPydantic automatically handles nested models. Field paths use dot notation (`address.street`), and all levels are optimized together.

---

## Step 1: Define Nested Models

```python
from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str = Field(description="Street")
    city: str = Field(description="City")
    zip_code: str = Field(description="ZIP code")

class Customer(BaseModel):
    name: str = Field(description="Name")
    address: Address = Field(description="Address")
```

---

## Step 2: Create Examples

Examples work the same way with nested structures:

```python
from dspydantic import Example

examples = [
    Example(
        text="Jane Smith, 456 Oak Ave, San Francisco, CA 94102",
        expected_output={
            "name": "Jane Smith",
            "address": {
                "street": "456 Oak Ave",
                "city": "San Francisco",
                "zip_code": "94102"
            }
        }
    ),
]
```

Or use Pydantic model instances:

```python
examples = [
    Example(
        text="Jane Smith, 456 Oak Ave, San Francisco, CA 94102",
        expected_output=Customer(
            name="Jane Smith",
            address=Address(
                street="456 Oak Ave",
                city="San Francisco",
                zip_code="94102"
            )
        )
    ),
]
```

---

## Step 3: Optimize

Optimization works automatically with nested models:

```python
import dspy
from dspydantic import Prompter

# Configure DSPy first
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="your-api-key"))

# Create and optimize
prompter = Prompter(model=Customer)
result = prompter.optimize(examples=examples)
```

The optimizer handles all nested levels automatically.

---

## Step 4: View Optimized Descriptions

Field paths use dot notation:

```python
print(result.optimized_descriptions)
# {
#     "name": "Optimized description for name",
#     "address": "Optimized description for address",
#     "address.street": "Optimized description for street",
#     "address.city": "Optimized description for city",
#     "address.zip_code": "Optimized description for zip_code",
# }
```

---

## Step 5: Use Optimized Prompter

Extract data with nested structure:

```python
customer = prompter.run("Jane Smith, 456 Oak Ave, San Francisco, CA 94102")
print(customer.name)            # Jane Smith
print(customer.address.street)  # 456 Oak Ave
print(customer.address.city)    # San Francisco
```

---

## Deeply Nested Models

Works with any level of nesting:

```python
class Country(BaseModel):
    name: str = Field(description="Country name")
    code: str = Field(description="Country code")

class Location(BaseModel):
    address: Address
    country: Country

class Company(BaseModel):
    name: str
    location: Location

# Field paths: location.address.street, location.country.name, etc.
```

---

## What Gets Optimized

| Level | Field Path | What Gets Optimized |
|-------|------------|-------------------|
| Top | `name` | Field description |
| Nested | `address` | Field description |
| Deep | `address.street` | Field description |
| All | - | System and instruction prompts |

All levels are optimized together to achieve accurate extraction.

---

## Tips

- Nested models are automatically handled — no special configuration needed
- Field paths use dot notation for nested fields
- All levels are optimized together
- Use [Include or Exclude Fields](include-exclude-fields.md) to selectively optimize nested fields

---

## See Also

- [Include or Exclude Fields](include-exclude-fields.md) — Exclude fields from evaluation
- [Extract Structured Data](../tutorials/extract-structured-data.md) — Complete optimization workflow
- [Configure Optimization Parameters](configure-optimizations.md) — Advanced optimization options
