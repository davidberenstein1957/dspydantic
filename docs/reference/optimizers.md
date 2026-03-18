# Optimizers Reference

Complete guide to DSPy optimizers used in DSPydantic.

## Auto-Selection

DSPydantic automatically selects an optimizer based on example count:

| Examples | Optimizer | Characteristics |
|----------|-----------|-----------------|
| 1-2 | MIPROv2 zero-shot | No examples needed, one-shot optimization |
| 3-19 | BootstrapFewShot | Few-shot learning, builds demonstrations |
| 20+ | BootstrapFewShotWithRandomSearch | More search, better with many examples |

---

## Optimizer Comparison

| Optimizer | Speed | Quality | API Calls | Best For |
|-----------|-------|---------|-----------|----------|
| **MIPROv2** | Fast | Fair | Low | Few examples |
| **BootstrapFewShot** | Medium | Good | Medium | Standard use |
| **BootstrapFewShotWithRandomSearch** | Slow | Excellent | High | Many examples, quality focus |
| **COPRO** | Slow | Excellent | High | Complex tasks |
| **GEPA** | Medium | Good | Medium | Balanced |
| **SIMBA** | Fast | Fair | Low | Fast iteration |
| **BetterTogether** | Slow | Excellent | High | Ensemble |
| **BootstrapFinetune** | Very Slow | Excellent | Very High | Fine-tuning |

---

## When to Use Each

**Development/Quick Iteration:**
- MIPROv2 (fastest)
- SIMBA (fast)
- BootstrapFewShot (balanced)

**Production/Quality Focus:**
- BootstrapFewShotWithRandomSearch (recommended)
- COPRO (excellent results)
- BetterTogether (ensemble)

**Fine-tuning:**
- BootstrapFinetune (creates training data)

---

## Configuration

Default configuration (auto-selected):

```python
result = prompter.optimize(examples=examples)
# Automatically picks best optimizer for example count
```

Manual selection:

```python
result = prompter.optimize(
    examples=examples,
    optimizer="BootstrapFewShot"
)
```

---

## API Call Estimates

For 10 examples with a 5-field model:

| Optimizer | Baseline | Optimization | Total |
|-----------|----------|--------------|-------|
| MIPROv2 | ~2 | ~15 | ~17 |
| BootstrapFewShot | ~2 | ~40 | ~42 |
| BootstrapFewShotWithRandomSearch | ~2 | ~80 | ~82 |
| COPRO | ~2 | ~120 | ~122 |

Actual costs depend on model, field count, and optimization duration.

---

## Performance Tips

**Reduce API calls:**
- Use fewer examples
- Use `fast=True`
- Use a cheaper model during optimization
- Use `include_fields` to focus on critical fields

**Improve quality:**
- Add more diverse examples
- Use a stronger model
- Use `fast=False`
- Choose quality-focused optimizer

---

## See Also

- [Configure Optimization Parameters](../how-to/configure-optimizations.md) — How-to guide
- [How Optimization Works](../explanation/how-optimization-works.md) — Technical details
