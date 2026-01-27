# 🚀 DSPydantic: Auto-Optimize Your Pydantic Models with DSPy

Automatically optimize Pydantic model field descriptions and prompts using DSPy. Get better structured data extraction from LLMs with less manual tuning.

## ✨ What It Does

Instead of spending hours crafting the perfect field descriptions for your Pydantic models, DSPydantic uses DSPy's optimization algorithms to automatically find the best descriptions based on your examples. Just provide a few examples, and watch your extraction accuracy improve.

<img width="1541" height="781" alt="Screenshot 2025-12-10 at 17 54 17" src="https://github.com/user-attachments/assets/c43a2cd0-1c49-417f-9775-5a51c3a6fb12" />

## 🎯 Quick Start

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Prompter, Example, create_optimized_model

# 1. Define your model (any Pydantic model works)
class TransactionRecord(BaseModel):
    broker: str = Field(description="Financial institution or brokerage firm")
    amount: str = Field(description="Transaction amount with currency")
    security: str = Field(description="Stock, bond, or financial instrument")
    date: str = Field(description="Transaction date")
    transaction_type: Literal["equity", "bond", "option", "future", "forex"] = Field(
        description="Type of financial instrument"
    )

# 2. Provide examples (just input text + expected output)
examples = [
    Example(
        text="Transaction Report: Goldman Sachs processed a $2.5M equity trade for Tesla Inc. on March 15, 2024.",
        expected_output=TransactionRecord(
            broker="Goldman Sachs",
            amount="$2.5M",
            security="Tesla Inc.",
            date="March 15, 2024",
            transaction_type="equity"
        )
    ),
    Example(
        text="JPMorgan executed $500K bond purchase for Apple Corp dated 2024-03-20.",
        expected_output=TransactionRecord(
            broker="JPMorgan",
            amount="$500K",
            security="Apple Corp",
            date="2024-03-20",
            transaction_type="bond"
        )
    ),
]

# 3. Optimize and use
prompter = Prompter(
    model=TransactionRecord,
    system_prompt="You are a financial document analysis assistant.",
    instruction_prompt="Extract transaction details from the financial report.",
)
result = prompter.optimize(examples=examples, model_id="gpt-4o")

OptimizedTransactionRecord = create_optimized_model(
    TransactionRecord,
    result.optimized_descriptions
)
print(result.optimized_descriptions)
print(result.optimized_system_prompt)
print(result.optimized_instruction_prompt)
# Use OptimizedTransactionRecord just like your original model, but with better accuracy!
```

**That's it!** Your model now has optimized descriptions that extract data more accurately.

## 📦 Installation

```bash
pip install dspydantic
```

Or with `uv`:

```bash
uv pip install dspydantic
```

## 🌟 Key Features

- **Auto-optimization**: Finds best field descriptions automatically
- **Unified Prompter class**: Single class for both optimization and extraction
- **Save & Load**: Save optimized prompters for production deployment
- **Pre-defined feedback**: Use pre-computed scores for evaluation
- **Simple input**: Just examples (text/images/PDFs) + your Pydantic model
- **Better output**: Optimized model ready to use with improved accuracy
- **Template prompts**: Dynamic prompts with `{placeholders}` for context-aware extraction
- **Enum & Literal support**: Optimize classification models
- **Multiple formats**: Text, images, PDFs—works with any input type
- **Smart defaults**: Auto-selects best optimizer, no configuration needed
- **Backward compatible**: `PydanticOptimizer` still available; `Prompter` is the recommended API

## 📚 Examples

Check out the [examples directory](examples/) for complete working examples:

- **[Veterinary EHR extraction](examples/text_example.py)**: Extract diseases, ICD-11 labels, and anonymized entities from clinical narratives—real-world medical data extraction
- **[Image classification](examples/image_example.py)**: Classify MNIST handwritten digits using `Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`—demonstrates vision capabilities and Literal type optimization
- **[Text classification](examples/imdb_example.py)**: Classify IMDB movie review sentiment with `Literal["positive", "negative"]` and template prompts—shows dynamic prompt formatting with `{review}` placeholders

## Basic Usage

### 1. Define Your Pydantic Model

```python
from pydantic import BaseModel, Field
from typing import Literal

class ProductInfo(BaseModel):
    name: str = Field(description="Full product name and model")
    storage: str = Field(description="Storage capacity like 256GB or 1TB")
    processor: str = Field(description="Chip or processor information")
    price: str = Field(description="Product price with currency")
    colors: list[str] = Field(description="Available color options")
    availability: Literal["in_stock", "pre_order", "sold_out"] = Field(
        description="Current availability status"
    )
```

### 2. Create Examples

**Simple input format**—just text + expected output:

```python
from dspydantic import Example

# Plain text input
examples = [
    Example(
        text="iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199. Available in titanium and black colors.",
        expected_output=ProductInfo(
            name="iPhone 15 Pro Max",
            storage="256GB",
            processor="A17 Pro chip",
            price="$1199",
            colors=["titanium", "black"],
            availability="in_stock"
        )
    ),
    Example(
        text="MacBook Air M3, 512GB SSD, M3 processor, $1299. Colors: space gray, silver. Currently on pre-order.",
        expected_output=ProductInfo(
            name="MacBook Air M3",
            storage="512GB SSD",
            processor="M3 processor",
            price="$1299",
            colors=["space gray", "silver"],
            availability="pre_order"
        )
    ),
]

# Or use dictionaries for template prompts (see Template Usage section)
# Or use images: Example(image_path="product.png", expected_output=...)
# Or use PDFs: Example(pdf_path="catalog.pdf", expected_output=...)
```

### 3. Optimize

```python
from dspydantic import Prompter

prompter = Prompter(model=ProductInfo)
result = prompter.optimize(examples=examples, model_id="gpt-4o")  # Returns optimized descriptions

# Access optimized results
result.optimized_descriptions        # dict[str, str] - optimized field descriptions
result.optimized_system_prompt      # str | None - optimized system prompt
result.optimized_instruction_prompt # str | None - optimized instruction prompt
```

**Template Formatting**: When using `text` as a dictionary, instruction prompt templates with placeholders like `{key}` are automatically formatted with values from each example's text dict. This allows you to create dynamic, example-specific prompts. See the [Template Usage](#template-usage-with-dynamic-prompts) section for a complete example.

### 4. Use Your Optimized Model

**Simple output**—just use the optimized model like your original:

```python
from dspydantic import create_optimized_model
from openai import OpenAI

# Create optimized model (drop-in replacement)
OptimizedProductInfo = create_optimized_model(
    ProductInfo,
    result.optimized_descriptions
)

# Use with OpenAI structured outputs
client = OpenAI()
messages = []

# Add optimized system prompt if available
if result.optimized_system_prompt:
    messages.append({
        "role": "system",
        "content": result.optimized_system_prompt
    })

# Prepare user content with optimized instruction prompt
user_content = (
    "Samsung Galaxy S24 Ultra, 1TB storage, Snapdragon 8 Gen 3 processor, "
    "$1299. Available in titanium black, titanium gray, and titanium violet. In stock now."
)
if result.optimized_instruction_prompt:
    user_content = f"{result.optimized_instruction_prompt}\n\n{user_content}"

messages.append({
    "role": "user",
    "content": user_content
})

# Call OpenAI API with optimized model
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format=OptimizedProductInfo
)

# Parse response using the optimized model
product = OptimizedProductInfo.model_validate_json(
    response.choices[0].message.content
)
```

**That's it!** Your optimized model extracts data more accurately with zero code changes.

## 🚀 Unified Prompter Workflow

Use the new `Prompter` class for a unified optimization and extraction workflow:

```python
from dspydantic import Prompter, Example
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Product price")

# Create prompter
prompter = Prompter(
    model=Product,
    model_id="gpt-4o"
)

# Optimize
result = prompter.optimize(
    examples=[
        Example(
            text="iPhone 15 Pro Max, $1199",
            expected_output={"name": "iPhone 15 Pro Max", "price": 1199.0}
        )
    ]
)

# Save for production
prompter.save("./production_prompter")

# Load in production
prompter = Prompter.load("./production_prompter", model=Product)

# Extract
data = prompter.extract("Samsung Galaxy S24, $899")
print(data.name)  # "Samsung Galaxy S24"
print(data.price)  # 899.0
```

## 💾 Save & Load Prompters

Save optimized prompters for production deployment:

```python
# After optimization
prompter.save("./production_prompter")

# Load anywhere (API keys must be available)
prompter = Prompter.load("./production_prompter", model=Product, api_key="your-key")

# Use directly
data = prompter.extract("New product text")
```

**What gets saved:**
- Complete Pydantic model schema
- All optimized field descriptions
- Optimized system and instruction prompts
- Model configuration (model_id, api_base, api_version)
- Optimization metadata

**Security:** API keys are NEVER saved - must be provided at load time.

## 📊 Pre-defined Feedback Evaluation

Use pre-computed scores for evaluation when you already have ground truth:

```python
from dspydantic import Prompter
from dspydantic.evaluators import PredefinedScoreEvaluator

# Pre-computed scores
scores = [0.95, 0.87, 0.92, 1.0, 0.78]
evaluator = PredefinedScoreEvaluator(config={"scores": scores})

prompter = Prompter(model=Product, model_id="gpt-4o")
result = prompter.optimize(
    examples=examples,
    evaluate_fn=evaluator  # Uses pre-defined scores
)
```

Works with bool values and numbers too:

```python
# Bool values (True=1.0, False=0.0)
bool_scores = [True, False, True, True]
evaluator = PredefinedScoreEvaluator(config={"scores": bool_scores})

# Numbers (normalized to 0.0-1.0)
numeric_scores = [95, 87, 92, 100]
evaluator = PredefinedScoreEvaluator(config={"scores": numeric_scores, "max_value": 100})
```

**Alternative: Python function that pops from list:**

```python
# Create a function that pops scores from a list
def pop_score_evaluator(example, optimized_descriptions, optimized_system_prompt, optimized_instruction_prompt):
    # Pre-defined scores list (shared state)
    if not hasattr(pop_score_evaluator, 'scores'):
        pop_score_evaluator.scores = [0.95, 0.87, 0.92, 1.0, 0.78]
    
    # Pop next score
    if pop_score_evaluator.scores:
        return pop_score_evaluator.scores.pop(0)
    return 0.0  # Default if list exhausted

prompter.optimize(examples=examples, evaluate_fn=pop_score_evaluator)
```

## 🏭 Real-World Usage Scenarios

### Financial Document Processing

```python
class Transaction(BaseModel):
    broker: str = Field(description="Financial institution")
    amount: str = Field(description="Transaction amount")
    security: str = Field(description="Financial instrument")
    transaction_type: Literal["equity", "bond", "option"] = Field(description="Transaction type")

examples = [
    Example(
        text="Goldman Sachs processed a $2.5M equity trade for Tesla Inc. on March 15, 2024.",
        expected_output=Transaction(broker="Goldman Sachs", amount="$2.5M", security="Tesla Inc.", transaction_type="equity")
    ),
    Example(
        text="JPMorgan executed $500K bond purchase for Apple Corp dated 2024-03-20.",
        expected_output=Transaction(broker="JPMorgan", amount="$500K", security="Apple Corp", transaction_type="bond")
    ),
]
```

### Healthcare Information Extraction

```python
from pydantic import BaseModel, Field
from dspydantic import Example

class MedicalRecord(BaseModel):
    patient_name: str = Field(description="Patient name")
    symptoms: list[str] = Field(description="Symptoms")
    medications: list[str] = Field(description="Prescribed medications")

examples = [
    Example(
        text="Patient: Sarah Johnson, 34. Symptoms: chest pain, shortness of breath. Prescribed: Lisinopril 10mg daily.",
        expected_output=MedicalRecord(
            patient_name="Sarah Johnson",
            symptoms=["chest pain", "shortness of breath"],
            medications=["Lisinopril 10mg daily"]
        )
    ),
    Example(
        text="Patient: Michael Chen, 45. Symptoms: headache, fatigue. Prescribed: Ibuprofen 400mg twice daily.",
        expected_output=MedicalRecord(
            patient_name="Michael Chen",
            symptoms=["headache", "fatigue"],
            medications=["Ibuprofen 400mg twice daily"]
        )
    ),
]
```

### Legal Contract Analysis

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example

class ContractAnalysis(BaseModel):
    parties: list[str] = Field(description="Contracting parties")
    effective_date: str = Field(description="Effective date")
    monthly_fee: str = Field(description="Monthly fee")
    contract_type: Literal["service", "employment", "nda"] = Field(description="Contract type")

examples = [
    Example(
        text="Service Agreement between TechCorp LLC and DataSystems Inc., effective January 1, 2024. Monthly fee: $15,000.",
        expected_output=ContractAnalysis(
            parties=["TechCorp LLC", "DataSystems Inc."],
            effective_date="January 1, 2024",
            monthly_fee="$15,000",
            contract_type="service"
        )
    ),
]
```

## Advanced Usage

### Other modalities

#### Working with Images

Just provide image paths + expected output:

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example

class DigitClassification(BaseModel):
    digit: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = Field(description="Digit 0-9")

examples = [
    Example(
        image_path="digit_5.png",
        expected_output=DigitClassification(digit=5)
    ),
    Example(
        image_path="digit_3.png",
        expected_output=DigitClassification(digit=3)
    ),
]
```

#### Working with PDFs

Just provide PDF paths + expected output:

```python
from pydantic import BaseModel, Field
from dspydantic import Example

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice number")
    total_amount: float = Field(description="Total amount of the invoice")
    vendor: str = Field(description="Vendor or supplier name")
    date: str = Field(description="Invoice date")

examples = [
    Example(
        pdf_path="invoice_001.pdf",
        expected_output=Invoice(
            invoice_number="INV-2024-001",
            total_amount=1234.56,
            vendor="Acme Corporation",
            date="2024-03-15"
        )
    ),
]
```

### Optimizing Prompt Templates

Optional `system_prompt` and `instruction_prompt` are optimized along with field descriptions. Use template prompts with placeholders that are automatically filled from example data dictionaries:

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example

class ProductReview(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Review sentiment")
    rating: int = Field(description="Rating 1-5")
    aspects: list[Literal["camera", "performance", "battery"]] = Field(description="Product aspects")

examples = [
    Example(
        text={"review": "Amazing camera quality and fast performance!", "product": "iPhone 15 Pro", "category": "smartphone"},
        expected_output=ProductReview(sentiment="positive", rating=4, aspects=["camera", "performance"])
    ),
    Example(
        text={"review": "Poor battery life and overpriced.", "product": "Samsung Galaxy S24", "category": "smartphone"},
        expected_output=ProductReview(sentiment="negative", rating=2, aspects=["battery"])
    ),
]

# Template prompts with {placeholders} are automatically filled from dict keys
prompter = Prompter(
    model=ProductReview,
    system_prompt="You are an expert analyst specializing in {category} reviews.",
    instruction_prompt="Analyze the {category} review about {product}: {review}",
)
result = prompter.optimize(examples=examples, model_id="gpt-4o")
# Access: result.optimized_system_prompt, result.optimized_instruction_prompt, result.optimized_descriptions
```

Placeholders like `{category}`, `{product}`, `{review}` are automatically filled from each example's text dictionary. Both prompts are optimized along with field descriptions.

### Working with Enums and Literals

`Literal` and `Enum` types work automatically and are taken into account for optimization.

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example

class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Review sentiment")
    aspects: list[Literal["camera", "performance", "battery", "display", "price"]] = Field(description="Product aspects")
    rating: int = Field(description="Rating 1-5")

examples = [
    Example(
        text="Great camera quality and amazing performance. Overall 4/5.",
        expected_output=ReviewAnalysis(sentiment="positive", aspects=["camera", "performance"], rating=4)
    ),
    Example(
        text="Poor display quality and overpriced. Not worth it. Rating: 2 stars.",
        expected_output=ReviewAnalysis(sentiment="negative", aspects=["display", "price"], rating=2)
    ),
]
```

### Excluding Fields from Evaluation

If you have fields that shouldn't affect the evaluation score (e.g., metadata, timestamps, or fields you're not optimizing), you can exclude them:

```python
from pydantic import BaseModel, Field

class PatientRecord(BaseModel):
    patient_name: str = Field(description="Patient full name")
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="Urgency level of the case"
    )
    diagnosis: str = Field(description="Primary diagnosis")
    metadata: str = Field(description="Internal metadata")  # Not important for evaluation
    timestamp: str = Field(description="Record timestamp")  # Not important for evaluation

prompter = Prompter(model=PatientRecord)
result = prompter.optimize(
    examples=examples,
    model_id="gpt-4o",
    exclude_fields=["metadata", "timestamp"],  # These fields won't affect scoring
)
```

Excluded fields will still be extracted by the model, but they won't be included in the evaluation score calculation. This is useful when you have fields that are not critical for optimization or that you don't want to optimize for.

### Nested Models

Nested models work automatically and are taken into account for optimization. Field paths like `"address.street"` are handled automatically:

```python
from pydantic import BaseModel, Field
from dspydantic import Example

class Address(BaseModel):
    street: str = Field(description="Street")
    city: str = Field(description="City")
    zip_code: str = Field(description="ZIP code")

class Customer(BaseModel):
    name: str = Field(description="Name")
    address: Address = Field(description="Address")

examples = [
    Example(
        text="Jane Smith, 456 Oak Ave, San Francisco, CA 94102",
        expected_output=Customer(
            name="Jane Smith",
            address=Address(street="456 Oak Ave", city="San Francisco", zip_code="94102")
        )
    ),
]
```

### Evaluation Options

#### Built-in Evaluation

Use built-in options: `"exact"` or `"levenshtein"`:

```python
from dspydantic import Prompter

prompter = Prompter(model=PatientRecord)
result = prompter.optimize(
    examples=examples,
    evaluate_fn="exact",  # or "levenshtein" for fuzzy matching
    model_id="gpt-4o",
)
```

#### Custom Evaluation Function

```python
from dspydantic import Prompter

def evaluate(example, optimized_descriptions, system_prompt, instruction_prompt) -> float:
    # Returns score 0.0 to 1.0
    return 0.85

prompter = Prompter(model=Customer)
result = prompter.optimize(examples=examples, evaluate_fn=evaluate, model_id="gpt-4o")
```

#### LLM Judge (No Expected Output)

When `expected_output` is `None`, use an LLM as a judge for unlabeled data:

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example, Prompter
import dspy

class Transaction(BaseModel):
    broker: str = Field(description="Financial institution")
    amount: str = Field(description="Transaction amount")
    security: str = Field(description="Financial instrument")
    transaction_type: Literal["equity", "bond", "option"] = Field(description="Transaction type")

examples = [
    Example(text="Goldman Sachs processed a $2.5M equity trade for Tesla Inc.", expected_output=None),
    Example(text="JPMorgan executed $500K bond purchase for Apple Corp.", expected_output=None),
]

# Uses model_id LLM as judge by default
prompter = Prompter(model=Transaction)
result = prompter.optimize(examples=examples, model_id="gpt-4o")

# Or use a separate judge LLM
judge_lm = dspy.LM("gpt-4", api_key="your-api-key")
prompter = Prompter(model=Transaction)
result = prompter.optimize(examples=examples, evaluate_fn=judge_lm, model_id="gpt-4o-mini")
```

### Optimizer Selection

Auto-selects optimizer based on dataset size, or specify manually:

- **Auto (default)**: `< 20 examples` → `BootstrapFewShot`, `>= 20 examples` → `BootstrapFewShotWithRandomSearch`
- **Manual**: Pass string (`"miprov2"`, `"gepa"`, `"copro"`, etc.) or Teleprompter instance

```python
from dspydantic import Prompter
from dspy.teleprompt import MIPROv2

# Auto-select (default)
prompter = Prompter(model=PatientRecord)
result = prompter.optimize(examples=examples, model_id="gpt-4o")

# Manual selection
prompter = Prompter(model=PatientRecord)
result = prompter.optimize(examples=examples, optimizer="miprov2", model_id="gpt-4o")

# Custom optimizer instance
custom_optimizer = MIPROv2(metric=my_metric, num_threads=8)
prompter = Prompter(model=PatientRecord)
result = prompter.optimize(examples=examples, optimizer=custom_optimizer, model_id="gpt-4o")
```

## API Reference

The API is organized into four areas. See the [full documentation](https://davidberenstein1957.github.io/dspydantic/) for details.

| Area | Description |
|------|-------------|
| **[Prompter](docs/reference/api/prompter.md)** | Unified optimization and prediction: `optimize()`, `predict()`, `save()`, `load()`, `from_optimization_result()` |
| **[Types](docs/reference/api/types.md)** | `Example`, `OptimizationResult`, `PrompterState` |
| **[Extractor](docs/reference/api/extractor.md)** | `extract_field_descriptions()`, `extract_field_types()`, `create_optimized_model()`, `apply_optimized_descriptions()` |
| **[Evaluators](docs/reference/api/evaluators.md)** | `exact`, `levenshtein`, `text_similarity`, `score_judge`, `label_model_grader`, `python_code`, `predefined_score` |

### Prompter

Unified class for optimization and prediction.

```python
prompter = Prompter(model=MyModel, model_id="gpt-4o")
result = prompter.optimize(examples=[...])
prompter.save("./my_prompter")
prompter = Prompter.load("./my_prompter", model=MyModel)
data = prompter.predict("New text")  # or predict(image_path=..., pdf_path=...)
```

### Types

- **Example** – `text` | `image_path` | `pdf_path` | dict, plus `expected_output`
- **OptimizationResult** – `optimized_descriptions`, `optimized_system_prompt`, `optimized_instruction_prompt`, `metrics`, `baseline_score`, `optimized_score`
- **PrompterState** – Serialized prompter (used by save/load)

### Extractor

- **create_optimized_model(model, optimized_descriptions)** – New Pydantic model class with optimized field descriptions
- **apply_optimized_descriptions(model, optimized_descriptions)** – JSON schema with optimized descriptions
- **extract_field_descriptions(model)** – Field path → description dict
- **extract_field_types(model)** – Field path → type info

### Evaluators

| Alias | Use case |
|-------|----------|
| `exact` | Exact string match |
| `levenshtein` | Fuzzy string match |
| `text_similarity` | Semantic similarity |
| `score_judge` | Numeric scores (LLM judge) |
| `label_model_grader` | Labels/categories (LLM judge) |
| `python_code` | Custom evaluation logic |
| `predefined_score` | Pre-computed scores |

## Backward Compatibility

`PydanticOptimizer` remains available for callers that prefer the standalone API. Prefer `Prompter` for new code:

```python
# Preferred: Prompter (unified optimize + predict)
from dspydantic import Prompter, Example, create_optimized_model
prompter = Prompter(model=MyModel)
result = prompter.optimize(examples=examples, model_id="gpt-4o")
OptimizedModel = create_optimized_model(MyModel, result.optimized_descriptions)

# Legacy: PydanticOptimizer (standalone optimizer only)
from dspydantic import PydanticOptimizer, create_optimized_model
optimizer = PydanticOptimizer(model=MyModel, examples=examples, model_id="gpt-4o")
result = optimizer.optimize()
OptimizedModel = create_optimized_model(MyModel, result.optimized_descriptions)
```

Convert `OptimizationResult` to `Prompter` with `Prompter.from_optimization_result(model, result)`.

## License

Apache 2.0

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
