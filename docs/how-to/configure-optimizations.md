# Configure Optimization Parameters

Fine-tune optimization behavior with parameters like sequential mode, early stopping, auto-generated prompts, and more.

## Quick Reference

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `sequential` | `False` | Field-by-field optimization vs. single-pass |
| `parallel_fields` | `True` | Parallelize fields in sequential mode |
| `num_threads` | `4` | Parallel evaluation threads |
| `include_fields` | All | Focus optimization on specific fields |
| `exclude_fields` | None | Skip fields from scoring |
| `optimizer` | Auto | Which DSPy optimizer to use |
| `early_stopping_patience` | `None` | Stop after N fields without improvement |
| `auto_generate_prompts` | `False` | Auto-create system/instruction prompts |
| `optimizer_kwargs` | `None` | Extra kwargs for optimizer constructor (e.g. `auto`, `num_candidates`) |
| `compile_kwargs` | `None` | Extra kwargs for DSPy compile (e.g. `num_trials`) |

---

## Single-Pass vs Sequential Mode

By default, DSPydantic uses **single-pass mode** (`sequential=False`): all fields are optimized together in one DSPy compile. Use `sequential=True` for field-by-field optimization — slower but better quality:

```python
# Single-pass (default): fast, lower API costs
result = prompter.optimize(examples=examples)

# Sequential: field-by-field for better quality
result = prompter.optimize(examples=examples, sequential=True)

# Sequential + parallel: best of both
result = prompter.optimize(examples=examples, sequential=True, parallel_fields=True)
```

**Use single-pass for:**
- Quick iterations during development
- When you have many examples
- Initial prototyping

**Use sequential for:**
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

## Early Stopping

In sequential mode, stop optimizing fields when scores plateau:

```python
result = prompter.optimize(
    examples=examples,
    sequential=True,
    early_stopping_patience=2,  # Stop after 2 fields without improvement
)
```

Fields are optimized deepest-first. If 2 consecutive fields show no improvement, the remaining fields are skipped. This can significantly reduce API costs when most fields already have good descriptions.

---

## Auto-Generate Prompts

Automatically create system and instruction prompts from your model:

```python
result = prompter.optimize(
    examples=examples,
    auto_generate_prompts=True,
)
```

This generates:
- **System prompt**: `"You are an expert at extracting structured {ModelName} data from text. Be precise and faithful to the source text."`
- **Instruction prompt**: `"Extract the following fields from the given text: field1, field2, .... Return only values that are explicitly stated or clearly implied."`

Existing prompts are preserved — auto-generation only fills in missing ones.

---

## Skip Optimization Phases

Skip specific optimization phases when you want to keep certain parts fixed:

```python
# Only optimize prompts, keep field descriptions as-is
result = prompter.optimize(
    examples=examples,
    skip_field_description_optimization=True,
)

# Only optimize field descriptions, skip prompt optimization
result = prompter.optimize(
    examples=examples,
    skip_system_prompt_optimization=True,
    skip_instruction_prompt_optimization=True,
)
```

---

## Custom Optimizer and Compile Arguments

Pass extra arguments to the DSPy optimizer's constructor and `compile()` method:

```python
# Control MiPROv2 constructor: disable auto-mode, set candidate count
result = prompter.optimize(
    examples=examples,
    optimizer_kwargs={"auto": None, "num_candidates": 3},
    compile_kwargs={"num_trials": 5, "minibatch": False},
)
```

- **`optimizer_kwargs`**: Passed to the optimizer constructor (e.g., `auto`, `num_candidates` for MiPROv2)
- **`compile_kwargs`**: Passed to the `compile()` call (e.g., `num_trials`, `minibatch`)

This is useful for controlling MiPROv2's trial count, candidate generation, minibatch behavior, or other optimizer-specific parameters.

---

## Reducing API Costs

```python
# 1. Use fewer examples
result = prompter.optimize(examples=examples[:10])

# 2. Use single-pass mode (default)
result = prompter.optimize(examples=examples)

# 3. Use fewer threads
result = prompter.optimize(examples=examples, num_threads=2)

# 4. Include only critical fields
result = prompter.optimize(
    examples=examples,
    include_fields=["critical_field_1", "critical_field_2"]
)

# 5. Use early stopping in sequential mode
result = prompter.optimize(
    examples=examples,
    sequential=True,
    early_stopping_patience=2,
)

# 6. Limit optimizer trials
result = prompter.optimize(
    examples=examples,
    compile_kwargs={"num_trials": 5, "minibatch": False},
)

# 7. Use a cheaper model during optimization
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
result = prompter.optimize(examples=examples)
```

---

## Troubleshooting

**Optimization takes too long?**
- Use single-pass mode (default, `sequential=False`)
- Reduce example count
- Increase `num_threads`
- Use `include_fields` to focus on critical fields
- Use `compile_kwargs={"num_trials": 5}` to limit MiPROv2 trials

**API calls are too high?**
- Reduce example count
- Use single-pass mode
- Use `early_stopping_patience` in sequential mode
- Use a cheaper model
- Limit trials with `compile_kwargs`

**Accuracy is low?**
- Add more diverse examples
- Check that examples are correct
- Ensure field descriptions are specific
- Try `sequential=True` for field-by-field optimization
- Use `auto_generate_prompts=True` to add system/instruction prompts

---

## See Also

- [Configure Evaluators](configure-evaluators.md) — How to evaluate extraction quality
- [Reference: Optimizers](../reference/optimizers.md) — Deep dive into each optimizer
- [Extract Structured Data](../tutorials/extract-structured-data.md) — Complete workflow
