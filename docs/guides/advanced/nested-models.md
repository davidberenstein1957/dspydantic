# Nested Models

This guide shows you how to optimize Pydantic models with nested structures. DSPydantic automatically optimizes field descriptions and prompts for all levels of nesting.

## When to Use Nested Models

| Use Case | Example | Benefit |
|----------|---------|---------|
| **Addresses** | `customer.address.street` | Hierarchical data organization |
| **Complex Data** | `company.location.address` | Multi-level accuracy |
| **Related Entities** | `order.items.product` | Structured relationships |

Use nested models when your data has hierarchical relationships.

## Problem

You have nested Pydantic models and want to optimize field descriptions **and prompts** for all levels of nesting.

## Solution

DSPydantic automatically handles nested models. Field paths use dot notation (e.g., `address.street`), and all levels are optimized together.

## Steps

### 1. Define Nested Models

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

### 2. Create Examples

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

Or with Pydantic models:

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

### 3. Optimize

Optimization works automatically with nested models:

```python
from dspydantic import Prompter

# Configure DSPy first
import dspy
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

prompter = Prompter(model=Customer)

result = prompter.optimize(examples=examples)
```

The optimization process optimizes field descriptions **and prompts** for all nested levels.

### 4. View Optimized Descriptions

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

### 5. Use Optimized Prompter

After optimization, extract data efficiently:

```python
# Extract data
customer = prompter.run("Jane Smith, 456 Oak Ave, San Francisco, CA 94102")
print(customer.name)
print(customer.address.street)
```

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
```

Field paths: `location.address.street`, `location.country.name`, etc.

## What Gets Optimized

| Level | Field Path | What Gets Optimized |
|-------|------------|-------------------|
| Top | `name` | Field description |
| Nested | `address` | Field description |
| Deep | `address.street` | Field description |
| All | - | System and instruction prompts |

All levels are optimized together to achieve accurate extraction.

## Tips

- Nested models are automatically handled - no special configuration needed
- Field paths use dot notation for nested fields
- All levels are optimized together
- See [Reference: Extractor](../../reference/api/extractor.md) for field path details

## See Also

- [Field Exclusion](field-exclusion.md) - Exclude fields from evaluation
- [Your First Optimization](../optimization/first-optimization.md) - Complete optimization workflow
- [Reference: Prompter](../../reference/api/prompter.md) - Complete API documentation
