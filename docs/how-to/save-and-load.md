# Save and Load a Prompter

Persist optimized prompters for production use.

## Save After Optimization

```python
from dspydantic import Prompter

# Create and optimize
prompter = Prompter(model=MyModel, model_id="openai/gpt-4o-mini")
result = prompter.optimize(examples=examples)

# Save the optimized prompter
prompter.save("./my_prompter")
```

**What is saved:**
- Optimized field descriptions
- Optimized system and instruction prompts
- Model configuration (field types, structure)

**What is NOT saved:**
- API keys (you provide these at load time)
- Examples
- Optimization history

---

## Load in Production

```python
from dspydantic import Prompter

# Load the optimized prompter
prompter = Prompter.load(
    "./my_prompter",
    model=MyModel,
    model_id="openai/gpt-4o-mini",
    # API key can come from environment variable
)

# Use it for extraction
result = prompter.run("input text")
```

Configure DSPy before loading if needed:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="your-key"))

prompter = Prompter.load(
    "./my_prompter",
    model=MyModel,
    model_id="openai/gpt-4o-mini"
)
```

---

## Versioning

Use versioning for managing multiple prompter versions:

```python
from datetime import datetime

# Timestamp versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
prompter.save(f"./prompters/{timestamp}/model")

# Semantic versioning
prompter.save("./prompters/v1.0.0/model")

# Latest symlink (optional, depends on your setup)
# ln -s v1.0.0 ./prompters/latest
```

---

## Model Upgrade Compatibility

**Safe changes** (don't require re-optimization):
- Adding new optional fields

**Incompatible changes** (require re-optimization):
- Changing field types
- Renaming fields
- Removing fields

---

## Tips

- Always test loaded prompters before production
- Use environment variables for API keys
- Version your prompters for easy rollback
- Document which model version each prompter uses
- Keep a record of optimization parameters used

---

## See Also

- [Deploy to Production](deploy-to-production.md) — Docker, FastAPI, Lambda patterns
- [Integrate with Applications](integrate-with-applications.md) — Real-world integration patterns
- [Extract Structured Data](../tutorials/extract-structured-data.md) — Complete workflow
