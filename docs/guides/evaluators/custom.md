# Custom Evaluators

This guide shows you how to create custom evaluators for specialized evaluation needs. Custom evaluators allow you to implement domain-specific evaluation logic that guides optimization.

## Custom vs Built-in Comparison

| Aspect | Built-in Evaluators | Custom Evaluators |
|--------|-------------------|------------------|
| **Setup** | Simple configuration | Requires implementation |
| **Flexibility** | Limited to predefined logic | Full control over logic |
| **Use Case** | Common evaluation patterns | Domain-specific rules |
| **Maintenance** | Maintained by library | You maintain |
| **Performance** | Optimized | Depends on implementation |

## When to Create Custom Evaluators

| Scenario | Solution |
|----------|----------|
| Business rules not covered by built-ins | Custom evaluator |
| Domain-specific thresholds | Custom evaluator |
| Complex multi-field evaluation | Custom evaluator |
| Simple variations of built-ins | Use built-in with config |

## Creating a Custom Evaluator

Implement the `BaseEvaluator` protocol:

```python
class MyCustomEvaluator:
    """Custom evaluator for specific business logic."""
    
    def __init__(self, config: dict) -> None:
        """Initialize with configuration."""
        self.threshold = config.get("threshold", 0.1)
        self.field_name = config.get("field_name", None)
    
    def evaluate(
        self,
        extracted: Any,
        expected: Any,
        input_data: dict | None = None,
        field_path: str | None = None,
    ) -> float:
        """
        Evaluate extracted value against expected value.
        
        Returns a score between 0.0 (no match) and 1.0 (perfect match).
        """
        # Custom evaluation logic
        if isinstance(extracted, (int, float)) and isinstance(expected, (int, float)):
            diff = abs(extracted - expected)
            if diff <= self.threshold:
                return 1.0
            return max(0.0, 1.0 - (diff / abs(expected)))
        
        # Default: exact match
        return 1.0 if extracted == expected else 0.0
```

## Using a Custom Evaluator

Use your custom evaluator in optimization:

```python
from dspydantic import Prompter

# Configure DSPy first
import dspy
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

prompter = Prompter(model=MyModel)

result = prompter.optimize(
    examples=examples,
    evaluator_config={
        "default": {
            "class": MyCustomEvaluator,
            "config": {"threshold": 0.05},
        },
    },
)
```

## Per-Field Custom Evaluators

Use different evaluators for different fields:

```python
class NameEvaluator:
    def __init__(self, config: dict) -> None:
        self.case_sensitive = config.get("case_sensitive", False)
    
    def evaluate(self, extracted, expected, input_data=None, field_path=None) -> float:
        if self.case_sensitive:
            return 1.0 if extracted == expected else 0.0
        return 1.0 if extracted.lower() == expected.lower() else 0.0

class RatingEvaluator:
    def __init__(self, config: dict) -> None:
        self.tolerance = config.get("tolerance", 1)
    
    def evaluate(self, extracted, expected, input_data=None, field_path=None) -> float:
        diff = abs(extracted - expected)
        if diff == 0:
            return 1.0
        elif diff <= self.tolerance:
            return 0.8
        return max(0.0, 1.0 - (diff / 5))

# Configure DSPy first
import dspy
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

prompter = Prompter(model=ProductReview)

result = prompter.optimize(
    examples=examples,
    evaluator_config={
        "default": "exact",
        "field_overrides": {
            "product_name": {
                "class": NameEvaluator,
                "config": {"case_sensitive": True},
            },
            "rating": {
                "class": RatingEvaluator,
                "config": {"tolerance": 1},
            },
        },
    },
)
```

## Python Code Evaluator

For quick custom logic, use a callable:

```python
def age_evaluator(extracted, expected, input_data=None, field_path=None):
    """Custom evaluation function for age field."""
    if field_path == "age":
        diff = abs(extracted - expected)
        if diff == 0:
            return 1.0
        elif diff <= 2:
            return 0.8
        else:
            return max(0.0, 1.0 - (diff / 10))
    # For other fields, use exact match
    return 1.0 if extracted == expected else 0.0

result = prompter.optimize(
    examples=examples,
    evaluator_config={
        "default": "exact",
        "field_overrides": {
            "age": {
                "type": "python_code",
                "config": {
                    "function": age_evaluator,
                },
            },
        },
    },
)
```

## Implementation Checklist

| Step | Action | Notes |
|------|--------|-------|
| 1 | Define evaluation logic | What makes a good match? |
| 2 | Implement `__init__` | Accept configuration |
| 3 | Implement `evaluate` | Return score 0.0-1.0 |
| 4 | Handle edge cases | None, empty strings, etc. |
| 5 | Test with examples | Validate behavior |
| 6 | Use in optimization | Configure evaluator |

## Best Practices

1. **Keep it simple**: Start with built-in evaluators
2. **Test thoroughly**: Validate custom evaluators with known examples
3. **Document logic**: Comment complex evaluation logic
4. **Handle edge cases**: Consider None values, empty strings, etc.
5. **Return valid scores**: Always return values between 0.0 and 1.0

## See Also

- [Configure Evaluators](configure.md) - How to set up evaluators
- [When to Use Which](selection.md) - Choose the right evaluator
- [Understanding Evaluators](../../concepts/evaluators.md) - Deep dive into evaluators
- [Reference: Evaluators](../../reference/api/evaluators.md) - Complete API documentation
