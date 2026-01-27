# How Optimization Works

This page explains how DSPydantic optimization works, why it's effective, and the design decisions behind it.

## How Optimization Works

DSPydantic uses DSPy's optimization algorithms to automatically improve field descriptions **and prompts**. The process works by:

1. **Testing Variations**: The prompter generates variations of field descriptions **and prompts**
2. **Evaluating Performance**: Each variation is tested against your examples using an evaluation function
3. **Selecting Best**: The prompter selects the descriptions and prompts that perform best
4. **Iterating**: The process repeats, refining the selections until convergence

## Optimization Flow

```mermaid
flowchart TD
    A[User Examples] --> B[Prompter]
    B --> C[DSPy Optimizer]
    C --> D[Generate Variations]
    D --> E[Test Descriptions]
    D --> F[Test Prompts]
    E --> G[Evaluators]
    F --> G
    G --> H[Score Results]
    H --> I[Select Best]
    I --> J[Optimized Prompter]
    J --> K[Extract Data]
```

## Why It Works

### The Problem with Manual Descriptions and Prompts

Writing good field descriptions and prompts is hard because:

- You need to anticipate how the LLM will interpret descriptions and prompts
- Different LLMs may interpret the same description or prompt differently
- Optimal descriptions and prompts depend on your specific data and use case
- Small wording changes can significantly affect extraction accuracy

### The Solution: Automated Optimization

DSPydantic solves this by:

- **Testing many variations**: The prompter tries many different phrasings automatically
- **Using your data**: Optimization is tailored to your specific examples
- **Measuring performance**: Each variation is evaluated objectively
- **Finding patterns**: The prompter learns what works best for your use case

## Design Decisions

### Why DSPy?

DSPy provides proven optimization algorithms that work well for prompt engineering. We leverage:

- **BootstrapFewShot**: For small datasets (< 20 examples)
- **BootstrapFewShotWithRandomSearch**: For larger datasets (≥ 20 examples)
- **MIPROv2**: For advanced optimization needs

### Optimization Algorithms

| Algorithm | When Used | Speed | Quality | Examples |
|-----------|-----------|-------|---------|----------|
| **BootstrapFewShot** | < 20 examples | Fast | Good | Quick iterations |
| **BootstrapFewShotWithRandomSearch** | ≥ 20 examples | Medium | Better | Production optimization |
| **MIPROv2** | Advanced needs | Slow | Best | Final optimization |

### Why Field Descriptions?

Field descriptions are the primary interface between your schema and the LLM. Optimizing them:

- Directly improves extraction accuracy
- Works with any LLM that supports structured outputs
- Is transparent and interpretable
- Can be version controlled and reviewed

### Why Prompts Too?

System and instruction prompts provide context. Optimizing them:

- Improves overall extraction quality
- Helps the LLM understand the task better
- Works synergistically with field descriptions
- Is essential for accurate extraction

## Trade-offs

### Advantages

- **Automatic**: No manual tuning required
- **Data-driven**: Based on your actual examples
- **Effective**: Typically improves accuracy significantly (10-30%)
- **Flexible**: Works with any Pydantic model
- **Comprehensive**: Optimizes both descriptions and prompts

### Limitations

- **Requires examples**: Need 5-20 examples for good results
- **Takes time**: Optimization can take several minutes
- **API costs**: Uses LLM API calls during optimization
- **Example quality matters**: Better examples lead to better optimization

## When to Use Optimization

Optimization is most valuable when:

- You have example data available (5-20 examples)
- Extraction accuracy is important
- You're willing to invest time upfront
- You want to improve over manual descriptions and prompts

Consider manual descriptions when:

- You have very few or no examples
- Speed is more important than accuracy
- You have domain expertise to write good descriptions
- You need immediate results

## Understanding Results

### Baseline Score

The baseline score shows how well your initial descriptions and prompts perform. This gives you a starting point.

### Optimized Score

The optimized score shows how well the optimized descriptions and prompts perform. This is what you'll get in production.

### Improvement

The improvement percentage shows the gain from optimization. Even small improvements (5-10%) can be significant in production.

### Optimized Descriptions and Prompts

The optimized descriptions and prompts are tailored to your use case. They may:

- Be more specific than your originals
- Include domain-specific terminology
- Emphasize important aspects
- Be longer or shorter depending on what works
- Work synergistically together

## Further Reading

- [Architecture](architecture.md) - System design details
- [Understanding Evaluators](evaluators.md) - How evaluation works
- [Your First Optimization](../guides/optimization/first-optimization.md) - Complete workflow
- [Reference: Prompter](../reference/api/prompter.md) - Technical details
