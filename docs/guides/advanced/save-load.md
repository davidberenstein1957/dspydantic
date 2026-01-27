# How to Save and Load Prompters

This guide shows you how to save optimized prompters for production deployment and load them later.

## Problem

You've optimized a prompter and want to save it for production use without re-running optimization.

## Solution

Use `Prompter.save()` to save and `Prompter.load()` to load optimized prompters.

## Saving a Prompter

After optimization, save the prompter:

```python
from dspydantic import Prompter, Example

prompter = Prompter(
    model=Product,
    model_id="gpt-4o",
)

# Optimize
result = prompter.optimize(examples=[...])

# Save
prompter.save("./production_prompter")
```

This saves:
- Complete Pydantic model schema
- All optimized field descriptions
- Optimized system and instruction prompts
- Model configuration (model_id, api_base, api_version)
- Optimization metadata

## Loading a Prompter

Load a saved prompter:

```python
from dspydantic import Prompter

prompter = Prompter.load(
    "./production_prompter",
    model=Product,  # Optional if model schema was saved
    api_key="your-key",  # Required - API keys are never saved
)
```

## Using a Loaded Prompter

Use the loaded prompter directly:

```python
# Extract data
data = prompter.run("New product text")
print(data.name)
print(data.price)
```

## Security Note

**API keys are NEVER saved** - you must provide them at load time:

```python
prompter = Prompter.load(
    "./production_prompter",
    model=Product,
    api_key=os.getenv("OPENAI_API_KEY"),  # Provide at load time
)
```

## Loading Without Model

If the model schema was saved, you can load without providing the model:

```python
prompter = Prompter.load(
    "./production_prompter",
    api_key="your-key",
)
```

However, you'll need to set the model for extraction:

```python
from my_models import Product

prompter.model = Product
data = prompter.run("text")
```

## Creating from Optimization Result

You can also create a prompter from an `OptimizationResult`:

```python
from dspydantic import Prompter

prompter = Prompter(model=Product, model_id="gpt-4o")
result = prompter.optimize(examples=examples)

# Prompter is already ready to use, but you can save it
prompter.save("./my_prompter")
```

## Tips

- Save prompters after successful optimization
- Version your saved prompters (e.g., `prompter_v1`, `prompter_v2`)
- Store API keys securely (environment variables, secrets management)
- Test loaded prompters before deploying to production
- See [Reference: Prompter](../reference/api/prompter.md) for all options

## See Also

- [Deploying to Production](deployment.md) - Production deployment guide
- [Your First Optimization](../optimization/first-optimization.md) - Complete optimization workflow
- [Reference: Prompter](../../reference/api/prompter.md) - Complete API documentation
