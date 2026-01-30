# Configure Optimizations

This guide covers how to configure optimization parameters, choose the right DSPy optimizer, and understand API call costs.

---

## Quick Reference

| Factor | Default | Recommended | Impact |
|--------|---------|-------------|--------|
| Examples | - | 10-20 | Quality |
| Threads | 4 | 4-8 | Speed |
| Optimizer | Auto | Based on dataset | Quality/Cost |

---

## Number of Examples

| Examples | Speed | Quality | API Calls |
|----------|-------|---------|-----------|
| **5-10** | Fast | Good | ~50-100 |
| **10-20** | Medium | Better | ~100-200 |
| **20+** | Slower | Best | ~200-500+ |

**Tips:**

- Start with 5-10 for prototyping
- Use 10-20 for production
- Ensure diverse examples covering edge cases

---

## DSPy Optimizers

DSPydantic uses DSPy optimizers under the hood. Understanding them helps you choose the right one.

### Auto-Selection Logic

DSPydantic auto-selects based on dataset size:

| Examples | Auto-Selected Optimizer |
|----------|------------------------|
| 1-2 | MIPROv2 (zero-shot mode) |
| 3-19 | BootstrapFewShot |
| 20+ | BootstrapFewShotWithRandomSearch |

---

### Optimizer Comparison

| Optimizer | Speed | Quality | API Calls | Best For |
|-----------|-------|---------|-----------|----------|
| **BootstrapFewShot** | Fast | Good | ~N | Prototyping, small datasets |
| **BootstrapFewShotWithRandomSearch** | Medium | Better | ~N×10 | Production, reliable results |
| **MIPROv2 (light)** | Medium | Better | ~50 | Quick production |
| **MIPROv2 (medium)** | Slow | Best | ~200 | Balanced quality/cost |
| **MIPROv2 (heavy)** | Slowest | Best | ~500+ | Maximum quality |
| **COPRO** | Medium | Good | ~M×K | Debugging, understanding prompts |
| **GEPA** | Medium | Good | ~20-100 | Complex reasoning, interpretable |
| **SIMBA** | Medium | Better | Variable | Large datasets (500+), batch |
| **BetterTogether** | Slowest | Best | Sum of all | Maximum quality, production |
| **Ensemble** | - | Best | N per input | Reliability, variance reduction |
| **BootstrapFinetune** | Slow | Best | Variable | Permanent model improvements |

---

### BootstrapFewShot

**Purpose:** Simple few-shot learning by sampling demonstrations from successful traces.

**Best for:** Small datasets (10-50 examples), quick prototyping.

**How it works:**

1. Runs your program on training examples
2. Collects traces of successful executions (based on metric)
3. Selects best demonstrations to include in prompts

**API calls:** ~N calls for N examples

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(max_bootstrapped_demos=4)

result = prompter.optimize(examples=examples, optimizer=optimizer)
```

---

### BootstrapFewShotWithRandomSearch

**Purpose:** BootstrapFewShot with multiple random seeds to find better demonstrations.

**Best for:** Medium datasets (50-200 examples), reliable results.

**How it works:**

1. Runs BootstrapFewShot multiple times with different seeds
2. Evaluates each configuration on validation set
3. Returns best configuration

**API calls:** ~N × num_candidate_programs

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    max_bootstrapped_demos=4,
    num_candidate_programs=10,  # More = more calls, better results
)

result = prompter.optimize(examples=examples, optimizer=optimizer)
```

---

### MIPROv2 (Production Recommended)

**Purpose:** Multi-step instruction and prompt optimization. The most sophisticated general optimizer.

**Best for:** Production optimization (50-500+ examples), maximum quality.

**How it works:**

1. **Bootstrapping stage:** Collects traces from running program
2. **Grounded proposal stage:** Uses LM to propose better instructions
3. **Discrete search stage:** Bayesian optimization to find best combination

**API calls:**

| Mode | Calls | When to Use |
|------|-------|-------------|
| `light` | ~50 | Quick optimization |
| `medium` | ~200 | Balanced |
| `heavy` | ~500+ | Maximum quality |

```python
from dspy.teleprompt import MIPROv2

# Light mode - faster, fewer calls
optimizer = MIPROv2(auto="light", num_threads=8)

# Medium mode - balanced
optimizer = MIPROv2(auto="medium", num_threads=8)

# Heavy mode - best quality
optimizer = MIPROv2(auto="heavy", num_threads=8)

result = prompter.optimize(examples=examples, optimizer=optimizer)
```

---

### COPRO (Coordinate Descent)

**Purpose:** Optimizes prompts by coordinate descent - changing one aspect at a time.

**Best for:** Understanding which prompt components matter most, debugging.

**How it works:**

1. Starts with initial prompt
2. Optimizes each "coordinate" (instruction, format) independently
3. Combines best settings

**API calls:** ~M × K (M = coordinates, K = options each)

```python
from dspy.teleprompt import COPRO

optimizer = COPRO(verbose=True)

result = prompter.optimize(examples=examples, optimizer=optimizer)
```

---

### GEPA (Reflective Prompt Evolution)

**Purpose:** Generative Evolution of Prompts and Adaptations. Iteratively refines prompts through self-reflection.

**Best for:** Complex reasoning tasks, interpretable improvements.

**How it works:**

1. Evaluates current prompt on examples
2. Reflects on failures and successes
3. Proposes prompt modifications
4. Iterates until convergence

**API calls:** ~20-100 depending on iterations

```python
from dspy.teleprompt import GEPA

optimizer = GEPA(num_iterations=10, verbose=True)

result = prompter.optimize(examples=examples, optimizer=optimizer)
```

---

### SIMBA

**Purpose:** Scalable Instruction Meta-prompting for Batch Adaptation.

**Best for:** Large datasets (500+ examples), batch processing.

**How it works:**

1. Creates meta-prompts that teach the LM about the task
2. Scales well to large datasets
3. Focuses on instruction quality

**API calls:** Variable, scales with dataset

```python
from dspy.teleprompt import SIMBA

optimizer = SIMBA()

result = prompter.optimize(examples=examples, optimizer=optimizer)
```

---

### BetterTogether

**Purpose:** Combines multiple optimizers for best results.

**Best for:** Maximum quality needed, production deployments.

**How it works:**

1. Runs multiple optimizers in sequence or parallel
2. Uses results from one optimizer to inform the next
3. Returns best combined result

**API calls:** Sum of all component optimizer calls

```python
from dspy.teleprompt import BetterTogether, BootstrapFewShot, MIPROv2

optimizer = BetterTogether(
    optimizers=[
        BootstrapFewShot(max_bootstrapped_demos=4),
        MIPROv2(auto="light"),
    ]
)

result = prompter.optimize(examples=examples, optimizer=optimizer)
```

---

### Ensemble

**Purpose:** Combines multiple optimized programs at inference time.

**Best for:** Maximum reliability, reducing variance, production systems.

**How it works:**

1. Runs input through multiple programs
2. Aggregates outputs (voting, averaging, etc.)
3. Returns consensus result

**API calls at inference:** N calls per input (one per ensemble member)

```python
from dspy import Ensemble

# After training multiple programs
ensemble = Ensemble(programs=[prog1, prog2, prog3], method="majority_vote")
result = ensemble(input_data)
```

---

### BootstrapFinetune

**Purpose:** Finetune model weights instead of just prompts.

**Best for:** 100+ high-quality examples, permanent model improvements.

**How it works:**

1. Generates training data from traces
2. Finetunes underlying LM on that data
3. Uses finetuned model for inference

**API calls:** Depends on training data size + finetuning

```python
from dspy.teleprompt import BootstrapFinetune

train_kwargs = {
    "use_peft": True,  # Enable LoRA
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "learning_rate": 2e-4,
}

optimizer = BootstrapFinetune(train_kwargs=train_kwargs, num_threads=8)

result = prompter.optimize(examples=examples, optimizer=optimizer)
```

**Requirements for LoRA:** `pip install transformers accelerate trl peft`

---

## Parallel Evaluation

Use multiple threads for faster optimization:

```python
result = prompter.optimize(
    examples=examples,
    num_threads=4,  # Parallel evaluation
)
```

| Threads | Speed | Use Case |
|---------|-------|----------|
| 1 | Baseline | Debugging |
| 2-4 | 2-3x faster | Development |
| 4-8 | 3-4x faster | Production |

---

## API Call Tracking

After optimization, check usage:

```python
result = prompter.optimize(examples=examples)

print(f"API calls: {result.api_calls}")
print(f"Tokens used: {result.total_tokens:,}")
print(f"Baseline: {result.baseline_score:.0%}")
print(f"Optimized: {result.optimized_score:.0%}")
```

---

## Common API Call Issues

### Issue 1: Hidden Calls in Metrics

Metrics that use LLMs add calls per evaluation:

```python
# BAD: This makes an LM call per evaluation!
def my_metric(example, pred, trace=None):
    judge = dspy.ChainOfThought(JudgeSignature)
    return judge(pred.output).score  # Hidden call!

# GOOD: Use simple comparison metrics when possible
def my_metric(example, pred, trace=None):
    return pred.output == example.expected_output
```

### Issue 2: Uncached Repeated Calls

Same inputs without caching = repeated API calls:

```python
# Enable caching to avoid duplicate calls
prompter = Prompter(
    model=MyModel,
    model_id="openai/gpt-4o-mini",
    cache=True,  # Prevents duplicate API calls
)
```

### Issue 3: Optimizer Training Calls

Optimizers make many calls during compilation:

```python
# MIPROv2 medium makes ~200 calls
optimizer = MIPROv2(auto="medium")

# Start with light for testing (~50 calls)
optimizer = MIPROv2(auto="light")
```

---

## Reducing API Costs

1. **Start small:** Use 5-10 examples initially
2. **Use caching:** `cache=True` prevents duplicate calls
3. **Choose optimizer wisely:** BootstrapFewShot for prototyping
4. **Use cheaper models:** `gpt-4o-mini` for optimization, `gpt-4o` for production
5. **Start with light mode:** `MIPROv2(auto="light")` before `"heavy"`

```python
prompter = Prompter(
    model=MyModel,
    model_id="openai/gpt-4o-mini",  # Cheaper model for optimization
    cache=True,
)

# Start light
result = prompter.optimize(
    examples=examples[:10],  # Fewer examples first
    optimizer=MIPROv2(auto="light"),
)

# If results are good, try more examples
result = prompter.optimize(
    examples=examples,
    optimizer=MIPROv2(auto="medium"),
)
```

---

## Advanced: Custom Optimizer Kwargs

Pass additional arguments to DSPy optimizers:

```python
result = prompter.optimize(
    examples=examples,
    optimizer="miprov2",
    optimizer_kwargs={
        "max_bootstrapped_demos": 8,
        "auto": "medium",
        "num_threads": 8,
    },
)
```

---

## Troubleshooting

### Optimization is slow

- Reduce examples (start with 5-10)
- Use `BootstrapFewShot` instead of random search
- Increase `num_threads`
- Use `MIPROv2(auto="light")` instead of `"heavy"`

### High API costs

- Use cheaper model (`gpt-4o-mini`)
- Enable caching (`cache=True`)
- Start with fewer examples
- Use simpler optimizer first

### Poor optimization results

- Add more diverse examples
- Try `MIPROv2(auto="medium")` for better quality
- Check that examples are correct
- Ensure examples cover edge cases

---

## See Also

- [Configure Models](configure-models.md) - Model configuration
- [Your First Optimization](../optimization/first-optimization.md) - Complete workflow
- [How Optimization Works](../../concepts/optimization.md) - Deep dive
- [DSPy Documentation](https://dspy.ai/) - Official DSPy docs
