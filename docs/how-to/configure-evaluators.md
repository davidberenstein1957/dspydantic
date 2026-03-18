# Configure Evaluators

Set up and customize evaluation metrics for optimization.

## Evaluator Types

| Type | Best For | Speed | Example |
|------|----------|-------|---------|
| `exact` | IDs, SKUs, exact text | Fast | `"ABC123"` |
| `levenshtein` | Names, addresses (minor variations) | Fast | `"John"` vs `"Jon"` |
| `text_similarity` | Descriptions, free-form text | Medium | Semantic closeness |
| `score_judge` | Ratings, scores | Medium | LLM judges quality |
| `label_model_grader` | Categories, classifications | Medium | Multi-category voting |
| `python_code` | Custom logic | Varies | Your function |
| `predefined_score` | Pre-computed scores | Fast | Use fixed scores |

---

## Basic Configuration

```python
from dspydantic import Prompter

prompter = Prompter(model=MyModel)

result = prompter.optimize(
    examples=examples,
    evaluator_config={
        "default": {"type": "exact"},
        "field_overrides": {
            "name": {"type": "levenshtein", "config": {"threshold": 0.8}},
            "description": {"type": "text_similarity"},
        }
    }
)
```

---

## Per-Field Configuration

Set different evaluators for different fields:

```python
evaluator_config = {
    "default": {"type": "exact"},
    "field_overrides": {
        # Exact match for IDs
        "id": {"type": "exact"},

        # Allow minor variations for names
        "name": {
            "type": "levenshtein",
            "config": {"threshold": 0.85}
        },

        # Semantic matching for descriptions
        "description": {
            "type": "text_similarity",
            "config": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "threshold": 0.7
            }
        },

        # LLM judges for complex fields
        "quality_assessment": {
            "type": "score_judge",
            "config": {
                "criteria": "Does the assessment match the original?",
                "temperature": 0.1
            }
        },
    }
}

result = prompter.optimize(examples=examples, evaluator_config=evaluator_config)
```

---

## Python Code Evaluator

Use a custom function for evaluation:

```python
# Define evaluation logic
def age_evaluator(extracted, expected, input_data, field_path):
    try:
        ext_age = int(extracted)
        exp_age = int(expected)
        if ext_age == exp_age:
            return 1.0
        elif abs(ext_age - exp_age) <= 2:  # Within 2 years
            return 0.7
        else:
            return 0.0
    except:
        return 0.0

# Use in configuration
evaluator_config = {
    "default": {"type": "exact"},
    "field_overrides": {
        "age": {
            "type": "python_code",
            "function": age_evaluator
        }
    }
}

result = prompter.optimize(examples=examples, evaluator_config=evaluator_config)
```

---

## Predefined Scores

Use pre-computed evaluation scores:

```python
evaluator_config = {
    "default": {
        "type": "predefined_score",
        "config": {
            "scores": [1.0, 0.9, 0.8, 0.7, 0.6]  # One per example
        }
    }
}
```

---

## Common Patterns

### Exact for IDs, Semantic for Text

```python
{
    "field_overrides": {
        "invoice_id": {"type": "exact"},
        "description": {"type": "text_similarity"},
        "amount": {"type": "exact"},
    }
}
```

### Multi-Level Matching

```python
{
    "field_overrides": {
        "category": {"type": "exact"},
        "name": {"type": "levenshtein", "config": {"threshold": 0.85}},
        "details": {"type": "text_similarity", "config": {"threshold": 0.75}},
    }
}
```

---

## Tips

- Start with `exact` for most fields
- Use `levenshtein` for names/addresses with minor variations
- Use `text_similarity` for descriptions
- Use LLM-based evaluators sparingly (they're slower and more expensive)
- Test your configuration with a small subset of examples first

---

## See Also

- [Build a Custom Evaluator](build-custom-evaluators.md) — Create domain-specific evaluators
- [Choosing an Evaluator](../explanation/choosing-an-evaluator.md) — Decision guide for evaluator selection
- [Reference: Evaluators](../reference/api/evaluators.md) — Complete API documentation
