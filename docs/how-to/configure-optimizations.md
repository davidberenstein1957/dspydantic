# Configure Optimization Parameters

Fine-tune optimization behavior with parameters like fast mode, parallel evaluation, and example count.

## Quick Reference

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `fast` | `False` | Single-pass optimization vs. sequential |
| `num_threads` | `4` | Parallel evaluation threads |
| `include_fields` | All | Focus optimization on specific fields |
| `exclude_fields` | None | Skip fields from scoring |
| `optimizer` | Auto | Which DSPy optimizer to use |

---

## Fast Mode

Optimize in a single pass instead of field-by-field:

```python
# Default mode (slower, better quality)
result = prompter.optimize(examples=examples, fast=False)

# Fast mode (quicker, decent quality)
result = prompter.optimize(examples=examples, fast=True)
```

**Use `fast=True` for:**
- Quick iterations during development
- When you have many examples (slow with default)
- Initial prototyping

**Use `fast=False` for:**
- Production optimization
- When quality matters
- When you have few examples (< 20)

---

## Parallel Evaluation

Speed up evaluation with multiple threads:

```python
result = prompter.optimize(
    examples=examples,
    num_threads=8  # Default is 4
)
```

Each thread evaluates examples in parallel. More threads = faster, but:
- Uses more API calls simultaneously
- May hit rate limits
- Requires more resources

---

## Example Count

More examples = better optimization, but slower:

| Count | Time | Quality | Cost |
|-------|------|---------|------|
| 5 | Fast | Fair | Low |
| 10-20 | Good | Better | Medium |
| 20+ | Slow | Best | High |

Recommended: **10-20 examples** for most tasks.

---

## Include/Exclude Fields

Focus optimization on specific fields:

```python
# Only optimize these fields
result = prompter.optimize(
    examples=examples,
    include_fields=["name", "email"]
)

# Optimize everything except these
result = prompter.optimize(
    examples=examples,
    exclude_fields=["metadata", "timestamp"]
)
```

See [Include or Exclude Fields](include-exclude-fields.md) for details.

---

## Optimizer Selection

DSPydantic auto-selects based on example count:

| Examples | Optimizer |
|----------|-----------|
| 1-2 | MIPROv2 (zero-shot) |
| 3-19 | BootstrapFewShot |
| 20+ | BootstrapFewShotWithRandomSearch |

For advanced control, see [Reference: Optimizers](../reference/optimizers.md).

---

## API Cost Tracking

Track API usage after optimization:

```python
result = prompter.optimize(examples=examples)

print(f"API calls: {result.api_calls}")
print(f"Total tokens: {result.total_tokens:,}")
print(f"Baseline score: {result.baseline_score:.0%}")
print(f"Optimized score: {result.optimized_score:.0%}")
```

---

## Reducing API Costs

```python
# 1. Use fewer examples
result = prompter.optimize(examples=examples[:10])

# 2. Use fast mode
result = prompter.optimize(examples=examples, fast=True)

# 3. Use fewer threads
result = prompter.optimize(examples=examples, num_threads=2)

# 4. Include only critical fields
result = prompter.optimize(
    examples=examples,
    include_fields=["critical_field_1", "critical_field_2"]
)

# 5. Use a cheaper model during optimization
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
result = prompter.optimize(examples=examples)
```

---

## Troubleshooting

**Optimization takes too long?**
- Use `fast=True`
- Reduce example count
- Increase `num_threads`
- Use `include_fields` to focus on critical fields

**API calls are too high?**
- Reduce example count
- Use `fast=True`
- Use a cheaper model
- Use fewer threads

**Accuracy is low?**
- Add more diverse examples
- Check that examples are correct
- Ensure field descriptions are specific

---

## See Also

- [Configure Evaluators](configure-evaluators.md) — How to evaluate extraction quality
- [Reference: Optimizers](../reference/optimizers.md) — Deep dive into each optimizer
- [Extract Structured Data](../tutorials/extract-structured-data.md) — Complete workflow
