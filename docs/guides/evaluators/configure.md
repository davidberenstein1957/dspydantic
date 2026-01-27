# Configure Evaluators

This guide shows you how to configure evaluators for different fields and use cases. Evaluators guide the optimization process by measuring how well extracted data matches expected data.

## Problem

You need different evaluation strategies for different fields, or want to use pre-computed scores instead of running evaluations during optimization.

## Solution

Use `evaluator_config` to configure evaluators per field or use `PredefinedScoreEvaluator` for pre-computed scores.

## Available Evaluator Types

| Evaluator | Alias | Use Case | Data Types | Speed |
|-----------|-------|----------|------------|-------|
| `exact` | `exact` | Precise values that must match exactly | Strings | Fast |
| `levenshtein` | `levenshtein` | Text with minor spelling/formatting differences | Strings | Fast |
| `text_similarity` | `text_similarity` | Text where meaning matters more than exact wording | Strings | Medium |
| `score_judge` | `score_judge` | Numeric scores needing quality assessment | Numbers | Slow |
| `label_model_grader` | `label_model_grader` | Classification labels needing context-aware evaluation | Labels/Categories | Slow |
| `python_code` | `python_code` | Custom evaluation logic for complex business rules | Any | Medium |
| `predefined_score` | `predefined_score` | Pre-computed scores (no evaluation needed) | Any | Fastest |

## Common Configuration Patterns

| Pattern | Configuration | Use Case |
|---------|--------------|----------|
| **Most Fields Exact** | `default: "exact"` | Most fields need exact matching |
| **Text with Variations** | `default: "levenshtein"` | Text fields may have typos |
| **Semantic Matching** | `default: "text_similarity"` | Meaning matters more than wording |
| **Mixed Strategy** | `default` + `field_overrides` | Different fields need different evaluators |

## Using Pre-defined Scores

If you already have scores, use `PredefinedScoreEvaluator`:

```python
from dspydantic import Prompter
from dspydantic.evaluators import PredefinedScoreEvaluator

# Pre-computed scores
scores = [0.95, 0.87, 0.92, 1.0, 0.78]
evaluator = PredefinedScoreEvaluator(config={"scores": scores})

# Configure DSPy first
import dspy
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

prompter = Prompter(model=MyModel)

result = prompter.optimize(
    examples=examples,
    evaluate_fn=evaluator,
)
```

## Per-Field Evaluator Configuration

Configure different evaluators for different fields:

```python
# Configure DSPy first
import dspy
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

prompter = Prompter(model=User)

result = prompter.optimize(
    examples=examples,
    evaluator_config={
        "default": {
            "type": "exact",
            "config": {"case_sensitive": False},
        },
        "field_overrides": {
            "name": {
                "type": "exact",
                "config": {"case_sensitive": True},  # Names must match exactly
            },
            "description": {
                "type": "text_similarity",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "threshold": 0.7,
                },
            },
            "rating": {
                "type": "score_judge",
                "config": {
                    "criteria": "Rate the quality of this rating on a scale of 0-1",
                    "temperature": 0.0,
                },
            },
        },
    },
)
```

## Configuration Examples Table

| Field Type | Evaluator | Configuration | Reason |
|------------|-----------|---------------|--------|
| ID, SKU | `exact` | `case_sensitive: True` | Must match exactly |
| Name | `exact` | `case_sensitive: False` | Case variations OK |
| Description | `text_similarity` | `threshold: 0.7` | Meaning matters |
| Rating | `score_judge` | Custom criteria | Context-aware |
| Age | `python_code` | Custom function | Business rules |

## Custom Evaluator Class

Create a custom evaluator:

```python
# Configure DSPy first
import dspy
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

class ThresholdEvaluator:
    """Custom evaluator that checks if values are within a threshold."""
    
    def __init__(self, config: dict) -> None:
        self.threshold = config.get("threshold", 0.1)
    
    def evaluate(
        self,
        extracted: float,
        expected: float,
        input_data: dict | None = None,
        field_path: str | None = None,
    ) -> float:
        """Check if extracted value is within threshold of expected."""
        diff = abs(extracted - expected)
        return 1.0 if diff <= self.threshold else max(0.0, 1.0 - (diff / expected))

prompter = Prompter(model=RatingModel)

result = prompter.optimize(
    examples=examples,
    evaluator_config={
        "default": {
            "class": ThresholdEvaluator,
            "config": {"threshold": 0.05},
        },
    },
)
```

## Python Code Evaluator

Use a callable for custom evaluation logic:

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

# Configure DSPy first
import dspy
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

prompter = Prompter(model=SimpleUser)

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

## Tips

- Use `default` for most fields, override specific fields as needed
- Pre-defined scores are fastest when you have ground truth
- Text similarity works well for semantic matching
- See [When to Use Which](../evaluators/selection.md) for evaluator selection guidance
- See [Reference: Evaluators](../../reference/api/evaluators.md) for all options

## See Also

- [When to Use Which](selection.md) - Choose the right evaluator
- [Custom Evaluators](custom.md) - Create custom evaluation logic
- [Understanding Evaluators](../../concepts/evaluators.md) - Deep dive into evaluators
- [Reference: Evaluators](../../reference/api/evaluators.md) - Complete API documentation
