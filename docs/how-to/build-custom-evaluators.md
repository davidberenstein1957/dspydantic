# Build a Custom Evaluator

Create domain-specific evaluation logic for your extraction tasks.

## When to Use Custom Evaluators

| Scenario | Best Evaluator |
|----------|-----------------|
| Simple exact matches | Built-in `exact` |
| Minor spelling variations | Built-in `levenshtein` |
| Semantic similarity | Built-in `text_similarity` |
| **Custom business logic** | **Your own class** |
| **Complex evaluation rules** | **Your own class** |

Build a custom evaluator when built-in evaluators don't handle your domain-specific requirements.

---

## Create a Custom Evaluator Class

Implement the evaluator protocol:

```python
class MyEvaluator:
    def __init__(self, config=None):
        self.config = config or {}

    def evaluate(self, extracted, expected, input_data, field_path):
        """
        Compare extracted value to expected value.

        Returns float between 0.0 (fail) and 1.0 (perfect).
        """
        if extracted == expected:
            return 1.0
        elif similar(extracted, expected):
            return 0.5
        else:
            return 0.0
```

### Parameters

- `extracted` — The value your model extracted
- `expected` — The expected/reference value from your example
- `input_data` — The original input (useful for context)
- `field_path` — The field being evaluated (e.g., `"address.street"`)

### Return Value

Float between `0.0` (completely wrong) and `1.0` (perfect match).

---

## Example: Custom Rating Evaluator

```python
class RatingEvaluator:
    """Evaluate numeric ratings with tolerance."""

    def __init__(self, config=None):
        self.tolerance = (config or {}).get("tolerance", 0.5)

    def evaluate(self, extracted, expected, input_data, field_path):
        try:
            ext_val = float(extracted)
            exp_val = float(expected)

            # Perfect match
            if ext_val == exp_val:
                return 1.0

            # Within tolerance
            if abs(ext_val - exp_val) <= self.tolerance:
                return 0.7

            # Too far off
            return 0.0
        except (TypeError, ValueError):
            return 0.0
```

---

## Using a Custom Evaluator

Pass your evaluator class in the `evaluator_config`:

```python
from dspydantic import Prompter

prompter = Prompter(model=MyModel)

result = prompter.optimize(
    examples=examples,
    evaluator_config={
        "default": {"type": "exact"},
        "field_overrides": {
            "rating": {"class": RatingEvaluator, "config": {"tolerance": 0.5}},
        }
    }
)
```

---

## Evaluator Protocol

Your class must implement:

```python
class CustomEvaluator:
    def __init__(self, config=None):
        """Initialize with optional configuration."""
        pass

    def evaluate(self, extracted, expected, input_data, field_path):
        """Return float 0.0-1.0 representing match quality."""
        pass
```

---

## Tips

- Keep evaluators simple and fast
- Test thoroughly with your data
- Return 1.0 only for perfect matches
- Return 0.0 for completely wrong extractions
- Use intermediate values (0.5) for partial correctness
- Handle exceptions gracefully

---

## See Also

- [Configure Evaluators](configure-evaluators.md) — Built-in evaluators and configuration
- [Reference: Evaluators](../reference/api/evaluators.md) — Complete API documentation
