# Choosing an Evaluator

Guide for selecting the right evaluator for your extraction task.

## Quick Decision Tree

**Are values expected to match exactly?**
- Yes → Use `exact`
- No → Next question

**Do you expect minor variations (typos, spelling)?**
- Yes → Use `levenshtein`
- No → Next question

**Do you care about meaning/semantic similarity?**
- Yes → Use `text_similarity`
- No → Next question

**Is evaluation complex/subjective?**
- Yes → Use `score_judge` or custom evaluator
- No → Use `exact`

---

## Evaluator Comparison

| Evaluator | Speed | Accuracy | Cost | Best For |
|-----------|-------|----------|------|----------|
| `exact` | Fast | Perfect for exact | Free | IDs, SKUs, exact text |
| `levenshtein` | Fast | Good | Free | Names, minor variations |
| `text_similarity` | Medium | Good | Free* | Descriptions, semantic |
| `score_judge` | Slow | Excellent | $$$ | Complex evaluation |
| Custom | Varies | Excellent | Free | Domain logic |

*Requires embeddings model, can be free or paid depending on setup.

---

## Field Type Guide

| Field Type | Evaluator | Example |
|------------|-----------|---------|
| ID / SKU | `exact` | `"INV-2024-001"` |
| Name / Address | `levenshtein` | `"John"` vs `"Jon"` |
| Email | `exact` | `"john@example.com"` |
| Description | `text_similarity` | Long text fields |
| Category | `exact` | `"urgent"`, `"low"` |
| Rating / Score | Custom | `4.5` vs `4.0` |

---

## Trade-offs

**Fast evaluators** (`exact`, `levenshtein`):
- ✅ No API calls
- ✅ Deterministic
- ❌ Can't understand meaning

**Semantic evaluators** (`text_similarity`):
- ✅ Understand meaning
- ✅ Flexible
- ❌ Slower
- ❌ May need embedding model

**LLM evaluators** (`score_judge`):
- ✅ Very flexible
- ✅ Can handle complex logic
- ❌ Expensive (API calls)
- ❌ Non-deterministic

---

## Common Patterns

### Most Fields Exact, Some Semantic
```python
{
    "field_overrides": {
        "id": {"type": "exact"},
        "name": {"type": "levenshtein"},
        "description": {"type": "text_similarity"},
    }
}
```

### Text Fields with Variations
```python
{
    "default": {"type": "levenshtein", "config": {"threshold": 0.85}},
    "field_overrides": {
        "id": {"type": "exact"},
    }
}
```

### Custom Business Logic
```python
{
    "field_overrides": {
        "amount": {
            "type": "python_code",
            "function": lambda ext, exp, _, __: 1.0 if abs(float(ext) - float(exp)) < 0.01 else 0.0
        }
    }
}
```

---

## Tips

- Start simple: use `exact` first
- Measure baseline accuracy before optimizing
- Only add complexity (semantic, LLM) if needed
- Test evaluators on your actual data
- Consider evaluation cost when choosing

---

## See Also

- [Configure Evaluators](../how-to/configure-evaluators.md) — Configuration guide
- [Build a Custom Evaluator](../how-to/build-custom-evaluators.md) — Create custom logic
