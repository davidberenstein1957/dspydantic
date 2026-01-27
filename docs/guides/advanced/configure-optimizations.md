# Configure Optimizations

This guide covers how to configure optimization parameters to improve optimization performance and results.

## Number of Examples

The number of examples you provide affects both optimization speed and quality:

| Examples | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| **5-10** | Fast | Good | Quick iterations |
| **10-20** | Balanced | Better | Production optimization |
| **20+** | Slower | Best | Final optimization |

### Best Practices

- Start with 5-10 examples for initial testing
- Use 10-20 examples for production optimization
- Add more examples (20+) for final refinement
- Ensure examples cover diverse cases and edge cases

## Parallel Evaluation

Use multiple threads for faster optimization:

```python
import dspy
from dspydantic import Prompter, Example

# Configure DSPy first
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

prompter = Prompter(model=MyModel)

result = prompter.optimize(
    examples=examples,
    num_threads=4,  # Parallel evaluation
)
```

### Performance Impact

| Threads | Speed Improvement | Use Case |
|---------|------------------|----------|
| **1** (default) | Baseline | Small datasets |
| **2-4** | 2-3x faster | Medium datasets |
| **4-8** | 3-4x faster | Large datasets |

## Optimizer Selection

DSPydantic auto-selects optimizers, but you can choose manually:

| Optimizer | Speed | Quality | Use Case |
|-----------|-------|---------|----------|
| `BootstrapFewShot` | Fast | Good | Small datasets (< 20 examples) |
| `BootstrapFewShotWithRandomSearch` | Medium | Better | Larger datasets (≥ 20 examples) |
| `MIPROv2` | Slow | Best | Advanced optimization needs |

### Using Custom Optimizers

```python
import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
from dspydantic import Prompter

# Configure DSPy first
lm = dspy.LM("openai/gpt-4o", api_key="your-api-key")
dspy.configure(lm=lm)

prompter = Prompter(model=MyModel)

# Faster for small datasets
optimizer = BootstrapFewShot()

# Better for larger datasets
optimizer = BootstrapFewShotWithRandomSearch(num_candidates=10)

result = prompter.optimize(
    examples=examples,
    optimizer=optimizer,
)
```

### When to Override Auto-Selection

- **Small datasets (< 20 examples)**: Use `BootstrapFewShot` for speed
- **Large datasets (≥ 20 examples)**: Use `BootstrapFewShotWithRandomSearch` for quality
- **Complex optimization needs**: Use `MIPROv2` for advanced scenarios

## Optimization Parameters

### Number of Candidates

Control how many variations are tested:

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    num_candidates=10,  # Test 10 variations (default: varies)
)

result = prompter.optimize(
    examples=examples,
    optimizer=optimizer,
)
```

| Candidates | Speed | Quality | Use Case |
|------------|-------|---------|----------|
| **5-10** | Fast | Good | Quick optimization |
| **10-20** | Medium | Better | Balanced approach |
| **20+** | Slow | Best | Thorough optimization |

### Max Iterations

Limit optimization iterations:

```python
optimizer = BootstrapFewShotWithRandomSearch(
    max_iterations=5,  # Stop after 5 iterations
)
```

## Performance Comparison

| Optimization | Before | After | Impact |
|--------------|--------|-------|--------|
| Parallel Evaluation | Sequential | 4 threads | 3-4x faster |
| Optimizer Selection | Auto | BootstrapFewShot | 2x faster (small datasets) |
| Number of Candidates | Default | Reduced | 1.5-2x faster |

## Tips

- Start with default settings and adjust based on results
- Use parallel evaluation (`num_threads=4`) for faster optimization
- Choose optimizer based on dataset size
- Monitor optimization time vs quality trade-offs
- See [Configure Models](configure-models.md) for model-related performance

## See Also

- [Configure Models](configure-models.md) - Configure DSPy models for optimization
- [Your First Optimization](../optimization/first-optimization.md) - Complete optimization workflow
- [Reference: Prompter](../../reference/api/prompter.md) - Complete API documentation
