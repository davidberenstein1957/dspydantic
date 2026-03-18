# Evaluator Selection Reference

Quick lookup tables for evaluator selection.

## Quick Selection Table

| If You Need To... | Use Evaluator | Why |
|---|---|---|
| Match exact values | `exact` | Perfect precision |
| Allow minor typos | `levenshtein` | Edit distance matching |
| Compare meaning | `text_similarity` | Semantic matching |
| Complex judgment | `score_judge` | LLM evaluates |
| Domain-specific logic | `python_code` / Custom | Full flexibility |
| Pre-computed scores | `predefined_score` | Fixed scores |

---

## By Field Type

| Field Type | Evaluator | Example |
|---|---|---|
| ID / SKU | `exact` | `INV-2024-001` |
| Email | `exact` | `john@example.com` |
| Phone | `exact` or `levenshtein` | `555-1234` |
| Name | `levenshtein` | `John` vs `Jon` |
| Address | `levenshtein` | Minor spelling differences |
| City | `levenshtein` | `San Francisco` vs `SF` |
| Category | `exact` | Fixed categories |
| Price | Custom | Tolerance-based |
| Description | `text_similarity` | Semantic meaning |
| Summary | `text_similarity` | Paraphrase matching |
| Date | `exact` or Custom | Format specific |
| Status | `exact` | Fixed values |

---

## Performance & Cost

| Evaluator | Speed | Cost | Comments |
|---|---|---|---|
| `exact` | ⚡ Fast | Free | No API calls |
| `levenshtein` | ⚡ Fast | Free | No API calls |
| `text_similarity` | ⚡ Fast | Free/Low | Needs embeddings model |
| `score_judge` | 🐢 Slow | $$$ | LLM call per example |
| `label_model_grader` | 🐢 Slow | $$$ | Multiple LLM calls |
| `python_code` | ⚡-🐢 | Free | Depends on function |
| Custom | ⚡-🐢 | Free | Depends on logic |

---

## Configuration Examples

### Strict (Most Exact)
```python
{
    "default": {"type": "exact"},
    "field_overrides": {}
}
```

### Flexible (Some Variations)
```python
{
    "default": {"type": "levenshtein", "config": {"threshold": 0.85}},
    "field_overrides": {
        "id": {"type": "exact"},
    }
}
```

### Semantic (Meaning Matters)
```python
{
    "default": {"type": "text_similarity", "config": {"threshold": 0.7}},
    "field_overrides": {
        "id": {"type": "exact"},
        "category": {"type": "exact"},
    }
}
```

### Mixed (Multi-Strategy)
```python
{
    "field_overrides": {
        "id": {"type": "exact"},
        "name": {"type": "levenshtein"},
        "description": {"type": "text_similarity"},
        "status": {"type": "exact"},
    }
}
```

---

## Accuracy vs Speed Trade-off

**Fast & Simple:**
- `exact` — Perfect for exact matches
- `levenshtein` — Good for minor variations
- No API calls

**Balanced:**
- `text_similarity` — Understands meaning
- Moderate speed
- One-time embedding model setup

**Best Quality:**
- `score_judge` — LLM judges quality
- Slower optimization
- Expensive (API calls)

---

## Tips

1. **Start with `exact`** — Baseline fastest
2. **Add `levenshtein`** — If typos matter
3. **Switch to `text_similarity`** — If meaning matters
4. **Use LLM evaluators sparingly** — They're expensive
5. **Test on your data** — Quality needs vary

---

## See Also

- [Configure Evaluators](../how-to/configure-evaluators.md) — How-to guide
- [Choosing an Evaluator](../explanation/choosing-an-evaluator.md) — Decision guide
- [Build a Custom Evaluator](../how-to/build-custom-evaluators.md) — Custom logic
