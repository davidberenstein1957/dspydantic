# 🚀 DSPydantic: Auto-Optimize Your Pydantic Models with DSPy

Automatically optimize Pydantic model field descriptions and prompts using DSPy. Get better structured data extraction from LLMs with less manual tuning.

## ✨ What It Does

Instead of spending hours crafting the perfect field descriptions for your Pydantic models, DSPydantic uses DSPy's optimization algorithms to automatically find the best descriptions based on your examples. Just provide a few examples, and watch your extraction accuracy improve.

## 🎯 Quick Start

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import PydanticOptimizer, Example, create_optimized_model

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
optimizer = PydanticOptimizer(
    model=TransactionRecord,
    examples=examples,
    model_id="gpt-4o",
    system_prompt="You are a financial document analysis assistant.",
    instruction_prompt="Extract transaction details from the financial report.",
)
result = optimizer.optimize() 

OptimizedTransactionRecord = create_optimized_model(
    TransactionRecord,
    result.optimized_descriptions
)
# Use OptimizedTransactionRecord just like your original model, but with better accuracy!
```

**That's it!** Your model now has optimized descriptions that extract data more accurately—typically 20-40% improvement in extraction accuracy.

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

optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,
    model_id="gpt-4o",
    exclude_fields=["metadata", "timestamp"],  # These fields won't affect scoring
)
result = optimizer.optimize()
```

Excluded fields will still be extracted by the model, but they won't be included in the evaluation score calculation. This is useful when you have fields that are not critical for optimization or that you don't want to optimize for.

## 📦 Installation

```bash
pip install dspydantic
```

Or with `uv`:

```bash
uv pip install dspydantic
```

## 🌟 Key Features

- **Auto-optimization**: Finds best field descriptions automatically—20-40% accuracy improvement
- **Simple input**: Just examples (text/images/PDFs) + your Pydantic model
- **Better output**: Optimized model ready to use with improved accuracy
- **Template prompts**: Dynamic prompts with `{placeholders}` for context-aware extraction
- **Enum & Literal support**: Optimize classification models—often 70% → 90%+ accuracy
- **Multiple formats**: Text, images, PDFs—works with any input type
- **Smart defaults**: Auto-selects best optimizer, no configuration needed

## 📚 Examples

Check out the [examples directory](examples/) for complete working examples:

- **[Veterinary EHR extraction](examples/text_example.py)**: Extract diseases, ICD-11 labels, and anonymized entities from clinical narratives—real-world medical data extraction
- **[Image classification](examples/image_example.py)**: Classify MNIST handwritten digits using `Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`—demonstrates vision capabilities and Literal type optimization
- **[Text classification](examples/imdb_example.py)**: Classify IMDB movie review sentiment with `Literal["positive", "negative"]` and template prompts—shows dynamic prompt formatting with `{review}` placeholders
- **[Human-in-the-loop](examples/hitl_example.py)**: Interactive evaluation with GUI—get human feedback during optimization

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
from dspydantic import PydanticOptimizer

optimizer = PydanticOptimizer(
    model=Invoice,
    examples=examples,
    model_id="gpt-4o"
)

result = optimizer.optimize()  # Returns optimized descriptions

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

## Working with Images

**Easy input**: Just provide image paths + expected output:

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example, PydanticOptimizer, create_optimized_model

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

optimizer = PydanticOptimizer(
    model=DigitClassification,
    examples=examples,
    model_id="gpt-4o"
)
result = optimizer.optimize()  # Gain: 75% → 94% accuracy

OptimizedDigit = create_optimized_model(
    DigitClassification,
    result.optimized_descriptions
)
```

## Working with PDFs

**Easy input**: Just provide PDF paths + expected output:

```python
examples = [
    Example(
        pdf_path="invoice_001.pdf",
        expected_output=Invoice(
            invoice_number="INV-2024-001",
            total_amount=1234.56
        )
    ),
]
```

## Template Usage with Dynamic Prompts

Use template prompts with placeholders that are automatically filled from example data dictionaries. This is perfect for creating dynamic, context-aware prompts:

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example, PydanticOptimizer, create_optimized_model

class ProductReview(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    rating: int = Field(description="Rating from 1 to 5")
    aspects: list[Literal["camera", "performance", "battery", "display", "price"]] = Field(
        description="Product aspects mentioned"
    )

# Input: dictionary with placeholders
examples = [
    Example(
        text={
            "review": "Amazing camera quality and fast performance, but battery drains quickly. Overall great phone!",
            "customer": "Sarah Chen",
            "product": "iPhone 15 Pro",
            "category": "smartphone"
        },
        expected_output=ProductReview(
            sentiment="positive",
            rating=4,
            aspects=["camera", "performance", "battery"]
        )
    ),
    Example(
        text={
            "review": "Overpriced and poor display quality. Not worth the money.",
            "customer": "Mike Johnson",
            "product": "Samsung Galaxy S24",
            "category": "smartphone"
        },
        expected_output=ProductReview(
            sentiment="negative",
            rating=2,
            aspects=["price", "display"]
        )
    ),
]

# Template prompt with {placeholders} - automatically filled from dict keys
optimizer = PydanticOptimizer(
    model=ProductReview,
    examples=examples,
    system_prompt="You are an expert product review analyst specializing in {category} reviews.",
    instruction_prompt="Analyze the {category} review from {customer} about {product}: {review}",
    model_id="gpt-4o"
)

result = optimizer.optimize()

# The optimizer will automatically format the prompt for each example:
# Example 1: "Analyze the smartphone review from Sarah Chen about iPhone 15 Pro: Amazing camera..."
# Example 2: "Analyze the smartphone review from Mike Johnson about Samsung Galaxy S24: Overpriced..."
```

**Output**: Each example gets a customized prompt automatically—no manual formatting needed! Perfect for multi-domain extraction where context matters.

## Working with Enums and Literals

Use `Literal` or `Enum` in your model—works automatically and is used to optimize the extraction process. Perfect for classification tasks like sentiment analysis, document types, or status fields.

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example, PydanticOptimizer, create_optimized_model

class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    aspects: list[Literal["camera", "performance", "battery", "display", "price"]] = Field(
        description="Product aspects mentioned in the review"
    )
    rating: int = Field(description="Rating from 1 to 5")

examples = [
    Example(
        text="Great camera quality and amazing performance, but terrible battery life. Overall 4/5.",
        expected_output=ReviewAnalysis(
            sentiment="positive",
            aspects=["camera", "performance", "battery"],
            rating=4
        )
    ),
    Example(
        text="Poor display quality and overpriced. Not worth it. Rating: 2 stars.",
        expected_output=ReviewAnalysis(
            sentiment="negative",
            aspects=["display", "price"],
            rating=2
        )
    ),
]

optimizer = PydanticOptimizer(
    model=ReviewAnalysis,
    examples=examples,
    system_prompt="You are an expert product review analyst.",
    instruction_prompt="Analyze the product review and extract sentiment, mentioned aspects, and rating.",
    evaluate_fn="exact",
    model_id="gpt-4o"
)

result = optimizer.optimize()

OptimizedReviewAnalysis = create_optimized_model(
    ReviewAnalysis,
    result.optimized_descriptions
)
```

**Output**: Optimized descriptions help distinguish between similar categories automatically—often improving classification accuracy from 70% to 90%+!

## 🏭 Real-World Usage Scenarios

### Financial Document Processing

Extract structured financial data from transaction reports, trade confirmations, and financial statements:

```python
from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import Example, PydanticOptimizer, create_optimized_model

class Transaction(BaseModel):
    broker: str = Field(description="Financial institution or brokerage firm")
    amount: str = Field(description="Transaction amount with currency")
    security: str = Field(description="Stock, bond, or financial instrument")
    date: str = Field(description="Transaction date")
    commission: str = Field(description="Fees or commission charged")
    status: str = Field(description="Transaction status")
    transaction_type: Literal["equity", "bond", "option", "future", "forex"] = Field(
        description="Type of financial instrument"
    )

examples = [
    Example(
        text="Transaction Report: Goldman Sachs processed a $2.5M equity trade for Tesla Inc. on March 15, 2024. Commission: $1,250. Status: Completed.",
        expected_output=Transaction(
            broker="Goldman Sachs",
            amount="$2.5M",
            security="Tesla Inc.",
            date="March 15, 2024",
            commission="$1,250",
            status="Completed",
            transaction_type="equity"
        )
    ),
    Example(
        text="JPMorgan executed $500K bond purchase for Apple Corp dated 2024-03-20. Fee: $500. Status: Pending settlement.",
        expected_output=Transaction(
            broker="JPMorgan",
            amount="$500K",
            security="Apple Corp",
            date="2024-03-20",
            commission="$500",
            status="Pending settlement",
            transaction_type="bond"
        )
    ),
]

optimizer = PydanticOptimizer(
    model=Transaction,
    examples=examples,
    system_prompt="You are a financial document analysis assistant specializing in transaction extraction.",
    instruction_prompt="Extract all transaction details from the financial report.",
    model_id="gpt-4o"
)

result = optimizer.optimize()
OptimizedTransaction = create_optimized_model(Transaction, result.optimized_descriptions)
```

### Healthcare Information Extraction

Extract patient information, prescriptions, and medical data from clinical notes:

```python
from pydantic import BaseModel, Field
from typing import Literal

class PatientInfo(BaseModel):
    name: str = Field(description="Patient full name")
    age: str = Field(description="Patient age")
    symptoms: list[str] = Field(description="Reported symptoms or complaints")

class Prescription(BaseModel):
    medication: str = Field(description="Drug or medication name")
    dosage: str = Field(description="Dosage amount and frequency")
    frequency: str = Field(description="How often to take the medication")

class MedicalRecord(BaseModel):
    patient_info: PatientInfo = Field(description="Patient information")
    prescriptions: list[Prescription] = Field(description="Prescribed medications")
    follow_up: str = Field(description="Follow-up appointment information")

examples = [
    Example(
        text=(
            "Patient: Sarah Johnson, 34, presented with acute chest pain and shortness of breath. "
            "Prescribed: Lisinopril 10mg daily, Metoprolol 25mg twice daily. "
            "Follow-up scheduled for next Tuesday."
        ),
        expected_output=MedicalRecord(
            patient_info=PatientInfo(
                name="Sarah Johnson",
                age="34",
                symptoms=["acute chest pain", "shortness of breath"]
            ),
            prescriptions=[
                Prescription(medication="Lisinopril", dosage="10mg", frequency="daily"),
                Prescription(medication="Metoprolol", dosage="25mg", frequency="twice daily")
            ],
            follow_up="next Tuesday"
        )
    ),
]

optimizer = PydanticOptimizer(
    model=MedicalRecord,
    examples=examples,
    system_prompt="You are a medical information extraction assistant.",
    instruction_prompt="Extract patient information, prescriptions, and follow-up details from the medical record.",
    model_id="gpt-4o"
)

result = optimizer.optimize()
OptimizedMedicalRecord = create_optimized_model(MedicalRecord, result.optimized_descriptions)
```

### Legal Contract Analysis

Extract structured information from legal contracts and service agreements:

```python
from pydantic import BaseModel, Field
from typing import Literal

class ContractTerms(BaseModel):
    parties: list[str] = Field(description="Contracting parties involved")
    effective_date: str = Field(description="Contract effective date")
    monthly_fee: str = Field(description="Monthly payment amount")
    term_length: str = Field(description="Contract duration")
    renewal: Literal["automatic", "manual", "none"] = Field(
        description="Renewal type"
    )
    termination_notice: str = Field(description="Termination notice requirements")

class ContractAnalysis(BaseModel):
    contract_type: Literal["service", "employment", "nda", "partnership"] = Field(
        description="Type of contract"
    )
    terms: ContractTerms = Field(description="Contract terms and conditions")

examples = [
    Example(
        text=(
            "Service Agreement between TechCorp LLC and DataSystems Inc., effective January 1, 2024. "
            "Monthly fee: $15,000. Contract term: 24 months with automatic renewal. "
            "Termination clause: 30-day written notice required."
        ),
        expected_output=ContractAnalysis(
            contract_type="service",
            terms=ContractTerms(
                parties=["TechCorp LLC", "DataSystems Inc."],
                effective_date="January 1, 2024",
                monthly_fee="$15,000",
                term_length="24 months",
                renewal="automatic",
                termination_notice="30-day written notice"
            )
        )
    ),
]

optimizer = PydanticOptimizer(
    model=ContractAnalysis,
    examples=examples,
    system_prompt="You are a legal document analysis assistant.",
    instruction_prompt="Extract contract type and all terms from the service agreement.",
    model_id="gpt-4o"
)

result = optimizer.optimize()
OptimizedContractAnalysis = create_optimized_model(ContractAnalysis, result.optimized_descriptions)
```

### Knowledge Graph Construction

Extract entities and relationships for building knowledge graphs:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Entity(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: Literal["person", "organization", "location", "product"] = Field(
        description="Type of entity"
    )

class Relation(BaseModel):
    subject: str = Field(description="Subject entity name")
    relation_type: Literal["founded", "acquired", "located_in", "works_for"] = Field(
        description="Type of relationship"
    )
    object: str = Field(description="Object entity name")

class KnowledgeGraph(BaseModel):
    entities: list[Entity] = Field(description="Extracted entities")
    relations: list[Relation] = Field(description="Extracted relationships")

examples = [
    Example(
        text=(
            "Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne, California. "
            "SpaceX acquired Swarm Technologies in 2021."
        ),
        expected_output=KnowledgeGraph(
            entities=[
                Entity(name="Elon Musk", entity_type="person"),
                Entity(name="SpaceX", entity_type="organization"),
                Entity(name="Hawthorne, California", entity_type="location"),
                Entity(name="Swarm Technologies", entity_type="organization"),
            ],
            relations=[
                Relation(subject="Elon Musk", relation_type="founded", object="SpaceX"),
                Relation(subject="SpaceX", relation_type="located_in", object="Hawthorne, California"),
                Relation(subject="SpaceX", relation_type="acquired", object="Swarm Technologies"),
            ]
        )
    ),
]

optimizer = PydanticOptimizer(
    model=KnowledgeGraph,
    examples=examples,
    system_prompt="You are a knowledge graph extraction assistant.",
    instruction_prompt="Extract all entities and their relationships from the text.",
    model_id="gpt-4o"
)

result = optimizer.optimize()
OptimizedKnowledgeGraph = create_optimized_model(KnowledgeGraph, result.optimized_descriptions)
```

## Nested Models

Nested models work automatically—no special handling needed.

```python
class Address(BaseModel):
    street: str = Field(description="Street")
    city: str = Field(description="City")

class Customer(BaseModel):
    name: str = Field(description="Name")
    address: Address = Field(description="Address")

examples = [
    Example(
        text="Jane Smith, 456 Oak Ave, San Francisco",
        expected_output=Customer(
            name="Jane Smith",
            address=Address(
                street="456 Oak Ave",
                city="San Francisco"
            )
        )
    ),
]

# Field paths automatically handled: "name", "address.street", "address.city"
```

## Custom Evaluation

Provide your own evaluation function if needed.

```python
def evaluate(
    example,
    optimized_descriptions,
    system_prompt,
    instruction_prompt
) -> float:
    # Your custom logic - returns score 0.0 to 1.0
    return 0.85

optimizer = PydanticOptimizer(
    model=Customer,
    examples=examples,
    evaluate_fn=evaluate,  # Use custom evaluation
    model_id="gpt-4o"
)
```

## Evaluation Without Expected Output (LLM Judge)

When you don't have ground truth expected outputs, you can use an LLM as a judge to evaluate the quality of extracted data. This is useful when:

- You have unlabeled data—optimize on real-world examples without manual labeling
- You want to evaluate based on quality rather than exact matching—useful for subjective or nuanced extractions
- You need more nuanced evaluation criteria—e.g., "is this a reasonable extraction?" rather than exact match

### Using Default LLM Judge

When `expected_output` is `None`, the optimizer automatically uses the same LLM as a judge. Perfect for unlabeled data or when you want quality-based evaluation:

```python
from pydantic import BaseModel, Field
from typing import Literal

class FinancialTransaction(BaseModel):
    broker: str = Field(description="Financial institution")
    amount: str = Field(description="Transaction amount")
    security: str = Field(description="Financial instrument")
    transaction_type: Literal["equity", "bond", "option", "future", "forex"] = Field(
        description="Type of transaction"
    )

examples = [
    Example(
        text=(
            "Transaction Report: Goldman Sachs processed a $2.5M equity trade for Tesla Inc. "
            "on March 15, 2024. Commission: $1,250. Status: Completed."
        ),
        expected_output=None  # No ground truth, uses LLM judge
    ),
    Example(
        text=(
            "JPMorgan executed $500K bond purchase for Apple Corp dated 2024-03-20. "
            "Fee: $500. Status: Pending settlement."
        ),
        expected_output=None
    ),
]

optimizer = PydanticOptimizer(
    model=FinancialTransaction,
    examples=examples,
    system_prompt="You are a financial document analysis assistant.",
    instruction_prompt="Extract transaction details from the financial report.",
    model_id="gpt-4o",  # This LLM will be used as judge
    api_key="your-api-key"
)

result = optimizer.optimize()
```

### Using a Separate Judge LLM

You can pass a different `dspy.LM` as `evaluate_fn` to use as a judge. Useful when you want a more powerful model for evaluation:

```python
import dspy
from pydantic import BaseModel, Field

class ContractTerms(BaseModel):
    parties: list[str] = Field(description="Contracting parties")
    effective_date: str = Field(description="Contract effective date")
    monthly_fee: str = Field(description="Monthly payment amount")

# Create a separate judge LM (e.g., GPT-4 for judging, GPT-4o-mini for optimization)
judge_lm = dspy.LM(
    "gpt-4",
    api_key="your-api-key"
)

examples = [
    Example(
        text=(
            "Service Agreement between TechCorp LLC and DataSystems Inc., effective January 1, 2024. "
            "Monthly fee: $15,000. Contract term: 24 months."
        ),
        expected_output=None
    ),
]

optimizer = PydanticOptimizer(
    model=ContractTerms,
    examples=examples,
    evaluate_fn=judge_lm,  # Pass dspy.LM as evaluate_fn
    model_id="gpt-4o-mini",  # This LLM is used for optimization (cheaper)
    api_key="your-api-key"
)

result = optimizer.optimize()
```

**Note**: When `expected_output` is `None`:

- If `evaluate_fn` is a `dspy.LM`, it will be used as the judge
- If `evaluate_fn` is a callable, it will be treated as a judge function (with `extracted_data` parameter)
- If `evaluate_fn` is `None` or a string ("exact", "levenshtein", "exact-hitl", "levenshtein-hitl"), the default LLM judge will be used

## Optimizing Prompts

Provide optional prompts or template prompts with placeholders—they'll be optimized too. This is especially powerful for domain-specific extraction:

```python
from pydantic import BaseModel, Field
from typing import Literal

class ProductInfo(BaseModel):
    name: str = Field(description="Product name")
    price: str = Field(description="Product price")
    features: list[str] = Field(description="Product features")
    availability: Literal["in_stock", "pre_order", "sold_out"] = Field(
        description="Availability status"
    )

examples = [
    Example(
        text="iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199. Available in titanium and black colors.",
        expected_output=ProductInfo(
            name="iPhone 15 Pro Max",
            price="$1199",
            features=["256GB storage", "A17 Pro chip", "titanium design"],
            availability="in_stock"
        )
    ),
]

optimizer = PydanticOptimizer(
    model=ProductInfo,
    examples=examples,
    system_prompt="You are a product information extraction assistant.",  # Optional
    instruction_prompt="Extract product details from: {product_description}",  # Optional template
    model_id="gpt-4o"
)

result = optimizer.optimize()
# Typical results: baseline 68% → optimized 91% accuracy

# Create optimized model with updated descriptions
from dspydantic import create_optimized_model

OptimizedProductInfo = create_optimized_model(
    ProductInfo,
    result.optimized_descriptions
)

# Access optimized prompts
print(result.optimized_system_prompt)      # Optimized system prompt
print(result.optimized_instruction_prompt)  # Optimized instruction prompt
print(result.optimized_descriptions)        # Optimized field descriptions
```

## Built-in Evaluation Options

Instead of writing a custom evaluation function, you can use built-in options:

- `"exact"`: Exact matching between extracted and expected values
- `"levenshtein"`: Fuzzy matching using Levenshtein distance
- `"exact-hitl"`: Human-in-the-loop exact evaluation (shows GUI popup)
- `"levenshtein-hitl"`: Human-in-the-loop Levenshtein evaluation (shows GUI popup)

```python
optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,
    evaluate_fn="exact",  # or "levenshtein" for fuzzy matching
    model_id="gpt-4o"
)
```

## Optimizer Selection

DSPydantic supports multiple DSPy optimizers and automatically selects the best one based on your dataset size, or you can specify one manually.

### Auto-Selection (Default)

If you don't specify an optimizer, DSPydantic will automatically select one based on your dataset size:

- **< 20 examples**: Uses `BootstrapFewShot` (good for small datasets)
- **>= 20 examples**: Uses `BootstrapFewShotWithRandomSearch` (better for larger datasets)

```python
# Automatically selects best optimizer based on dataset size
optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,  # Will auto-select based on len(examples)
    model_id="gpt-4o"
)
```

### Manual Optimizer Selection

You can specify an optimizer by passing it as a string (optimizer type name) or as a Teleprompter instance:

```python
# Use a specific optimizer type (as string)
optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,
    optimizer="miprov2",  # or "gepa", "copro", "simba", etc.
    model_id="gpt-4o"
)
```

Available optimizer types include:

- `"bootstrapfewshot"`: BootstrapFewShot optimizer
- `"bootstrapfewshotwithrandomsearch"`: BootstrapFewShotWithRandomSearch
- `"miprov2"`: MIPROv2 optimizer (best for instruction optimization)
- `"miprov2zeroshot"`: MIPROv2 with zero-shot settings
- `"gepa"`: GEPA optimizer (reflective prompt evolution)
- `"copro"`: COPRO optimizer
- `"simba"`: SIMBA optimizer
- `"knnfewshot"`: KNNFewShot optimizer
- `"labeledfewshot"`: LabeledFewShot optimizer
- And more (all Teleprompter subclasses are automatically supported)

### Custom Optimizer Instance

You can also pass a custom optimizer instance directly:

```python
from dspy.teleprompt import MIPROv2

# Create a custom optimizer with specific settings
custom_optimizer = MIPROv2(
    metric=my_metric_function,
    num_threads=8,
    auto="full",
    max_bootstrapped_demos=8
)

# Use the custom optimizer
optimizer = PydanticOptimizer(
    model=PatientRecord,
    examples=examples,
    optimizer=custom_optimizer  # Pass custom optimizer instance
)
```

**Note**: The `optimizer` parameter accepts either a string (optimizer type name) or a Teleprompter instance. If None, it will auto-select based on dataset size.

## Examples

See the [examples directory](examples/) for complete working examples:

- **[Veterinary EHR extraction](examples/text_example.py)**: Extract diseases, ICD-11 labels, and anonymized entities from clinical narratives—real-world medical data extraction
- **[Handwritten digit classification](examples/image_example.py)**: Classify MNIST handwritten digits from images—demonstrates vision capabilities
- **[Movie review sentiment](examples/imdb_example.py)**: Classify IMDB movie review sentiment with template prompts—shows dynamic prompt formatting

## API Reference

### `PydanticOptimizer`

Main optimizer class.

**Parameters:**

- `model` (type[BaseModel]): The Pydantic model class to optimize—your structured data schema
- `examples` (list[Example]): List of examples for optimization—typically 5-20 examples yield good results
- `evaluate_fn` (Callable | dspy.LM | str | None): Evaluation function, built-in option ("exact", "levenshtein", "exact-hitl", "levenshtein-hitl"), or dspy.LM instance.
  - When `expected_output` is provided: Can be a callable `(Example, dict[str, str], str | None, str | None) -> float`,
    a string ("exact" or "levenshtein"), or None (uses default evaluation).
  - When `expected_output` is None: Can be a `dspy.LM` instance (used as judge), a callable judge function
    `(Example, dict[str, Any], dict[str, str], str | None, str | None) -> float`, or None (uses default LLM judge).
- `system_prompt` (str | None): Optional initial system prompt to optimize—helps set context for your domain
- `instruction_prompt` (str | None): Optional initial instruction prompt to optimize—can include template placeholders like `{key}`
- `lm` (dspy.LM | None): Optional DSPy language model instance. If provided, used instead of creating one from model_id/api_key.
- `model_id` (str): LLM model ID (default: "gpt-4o")—use "gpt-4o-mini" for faster/cheaper optimization
- `api_key` (str | None): API key (default: from OPENAI_API_KEY env var)
- `api_base` (str | None): API base URL (for Azure OpenAI or custom endpoints)
- `api_version` (str | None): API version (for Azure OpenAI)
- `num_threads` (int): Number of optimization threads (default: 4)—increase for faster optimization
- `init_temperature` (float): Initial temperature for optimization (default: 1.0)
- `verbose` (bool): Print progress (default: False)—set True to see optimization progress and scores
- `optimizer` (str | Teleprompter | None): Optimizer specification. Can be:
  - A string (optimizer type name): e.g., "miprov2", "gepa", "bootstrapfewshot", etc.
    If None, optimizer will be auto-selected based on dataset size.
  - A Teleprompter instance: Custom optimizer instance to use directly.
  Valid optimizer type strings include: "bootstrapfewshot", "bootstrapfewshotwithrandomsearch",
  "miprov2", "gepa", "copro", "simba", etc. (all Teleprompter subclasses are supported)
- `train_split` (float): Fraction of examples to use for training (rest for validation) (default: 0.8)
- `optimizer_kwargs` (dict[str, Any] | None): Optional dictionary of additional keyword arguments
  to pass to the optimizer constructor. Only used if `optimizer` is a string or None.

**Returns:**

- `OptimizationResult`: Contains optimized descriptions, prompts, and metrics

### `Example`

Example data for optimization.

**Parameters:**

- `expected_output` (dict | BaseModel | None): Expected output as a Pydantic model instance or dict.
  If `None`, evaluation will use an LLM judge or custom evaluation function instead of comparing
  against expected output.
- `text` (str | None): Plain text input
- `image_path` (str | Path | None): Path to an image file
- `image_base64` (str | None): Base64-encoded image string
- `pdf_path` (str | Path | None): Path to a PDF file
- `pdf_dpi` (int): DPI for PDF conversion (default: 300)

**Example:**

```python
from pydantic import BaseModel, Field
from typing import Literal

class Transaction(BaseModel):
    broker: str = Field(description="Financial institution")
    amount: str = Field(description="Transaction amount")
    security: str = Field(description="Financial instrument")
    transaction_type: Literal["equity", "bond", "option"] = Field(
        description="Transaction type"
    )

# Text input
Example(
    text="Goldman Sachs processed a $2.5M equity trade for Tesla Inc. on March 15, 2024.",
    expected_output=Transaction(
        broker="Goldman Sachs",
        amount="$2.5M",
        security="Tesla Inc.",
        transaction_type="equity"
    )
)

# Image input (e.g., scanned financial document)
Example(
    image_path="transaction_report.png",
    expected_output=Transaction(
        broker="JPMorgan",
        amount="$500K",
        security="Apple Corp",
        transaction_type="bond"
    )
)

# PDF input (e.g., financial statement PDF)
Example(
    pdf_path="financial_statement.pdf",
    expected_output=Transaction(
        broker="Morgan Stanley",
        amount="$1M",
        security="Microsoft Corp",
        transaction_type="equity"
    )
)

# Combined text and image
Example(
    text="Extract transaction details from this financial report",
    image_path="trade_confirmation.png",
    expected_output=Transaction(
        broker="Goldman Sachs",
        amount="$2.5M",
        security="Tesla Inc.",
        transaction_type="equity"
    )
)

# Template prompt with dictionary input
Example(
    text={
        "report": "Goldman Sachs processed a $2.5M equity trade for Tesla Inc.",
        "date": "March 15, 2024",
        "document_type": "transaction report"
    },
    expected_output=Transaction(
        broker="Goldman Sachs",
        amount="$2.5M",
        security="Tesla Inc.",
        transaction_type="equity"
    )
)

# Without expected_output (uses LLM judge for evaluation)
Example(
    text="JPMorgan executed $500K bond purchase for Apple Corp dated 2024-03-20.",
    expected_output=None
)
```

### `create_optimized_model(model, optimized_descriptions)`

**Recommended:** Create a new Pydantic model class with optimized descriptions.

**Parameters:**

- `model` (type[BaseModel]): Your original Pydantic model
- `optimized_descriptions` (dict[str, str]): From `result.optimized_descriptions`

**Returns:**

- `type[BaseModel]`: New model class with optimized descriptions in Field definitions

**Example:**

```python
from dspydantic import create_optimized_model
from pydantic import BaseModel, Field
from typing import Literal

class Transaction(BaseModel):
    broker: str = Field(description="Financial institution")
    amount: str = Field(description="Transaction amount")
    security: str = Field(description="Financial instrument")
    transaction_type: Literal["equity", "bond", "option"] = Field(
        description="Transaction type"
    )

# Create optimized model with improved field descriptions
OptimizedTransaction = create_optimized_model(
    Transaction,
    result.optimized_descriptions
)

# Use with OpenAI structured outputs
# The optimized descriptions improve extraction accuracy
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format=OptimizedTransaction
)
```

### `apply_optimized_descriptions(model, optimized_descriptions)`

Get optimized JSON schema without creating a new model class. Useful for one-off schema generation.

**Parameters:**

- `model` (type[BaseModel]): Your original Pydantic model
- `optimized_descriptions` (dict[str, str]): From `result.optimized_descriptions`

**Returns:**

- `dict`: JSON schema dictionary with optimized descriptions

**Example:**

```python
from dspydantic import apply_optimized_descriptions
from pydantic import BaseModel, Field
from typing import Literal

class ProductInfo(BaseModel):
    name: str = Field(description="Product name")
    price: str = Field(description="Product price")
    availability: Literal["in_stock", "pre_order", "sold_out"] = Field(
        description="Availability status"
    )

# Get optimized schema directly without creating a new model class
optimized_schema = apply_optimized_descriptions(
    ProductInfo,
    result.optimized_descriptions
)

# Use with OpenAI structured outputs
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "schema": optimized_schema
        }
    }
)
```

## License

Apache 2.0

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
