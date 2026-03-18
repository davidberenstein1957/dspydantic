# About Evaluators

This page explains the evaluation system in DSPydantic, how different evaluators work, and when to use each one.

## What Are Evaluators?

Evaluators measure how well extracted data matches expected data. They return scores between 0.0 (no match) and 1.0 (perfect match), which the prompter uses to find better descriptions.

## Evaluator Types Overview

| Evaluator | Alias | When to Use | Data Types | Speed | Accuracy |
|-----------|-------|-------------|------------|-------|----------|
| **StringCheckEvaluator** | `exact` | Precise values that must match exactly (IDs, codes, exact strings) | Strings | Fast | Exact |
| **LevenshteinEvaluator** | `levenshtein` | Text with minor spelling or formatting differences | Strings | Fast | Fuzzy |
| **TextSimilarityEvaluator** | `text_similarity` | Text where meaning matters more than exact wording | Strings | Medium | Semantic |
| **ScoreJudge** | `score_judge` | Numeric scores or ratings needing quality assessment | Numbers | Slow | LLM-based |
| **LabelModelGrader** | `label_model_grader` | Classification labels needing context-aware evaluation | Labels/Categories | Slow | LLM-based |
| **PythonCodeEvaluator** | `python_code` | Custom evaluation logic for complex business rules | Any | Medium | Custom |
| **PredefinedScoreEvaluator** | `predefined_score` | Pre-computed scores (no evaluation needed) | Any | Fastest | Pre-computed |

### Quick Selection Guide

- **Exact match needed?** → Use `exact` (StringCheckEvaluator)
- **Minor variations OK?** → Use `levenshtein` (LevenshteinEvaluator)
- **Semantic similarity?** → Use `text_similarity` (TextSimilarityEvaluator)
- **Complex evaluation?** → Use `score_judge` or `label_model_grader`
- **Custom logic?** → Use `python_code` (PythonCodeEvaluator)
- **Already have scores?** → Use `predefined_score` (PredefinedScoreEvaluator)

## Why Different Evaluators?

Different fields and use cases need different evaluation strategies:

- **Exact match**: For precise values like IDs or codes
- **Fuzzy match**: For text that may have minor variations
- **Semantic similarity**: For text with similar meaning but different wording
- **LLM-based**: For complex, context-dependent evaluation

## Built-in Evaluators

### Exact Evaluator (StringCheckEvaluator)

**When to use**: Precise values that must match exactly (IDs, codes, exact strings).

**How it works**: Compares extracted and expected values for exact equality.

**Example**: 
```python
# Product SKU must match exactly
extracted = "SKU-12345"
expected = "SKU-12345"
# Returns 1.0 if exact match, 0.0 otherwise
```

### Levenshtein Evaluator

**When to use**: Text that may have minor spelling or formatting differences.

**How it works**: Uses edit distance to measure similarity. Returns a score based on how many character changes are needed.

**Example**: 
```python
# Names with minor variations
extracted = "John Smith"
expected = "Jon Smith"  # Missing 'h'
# Returns ~0.9 (high similarity despite typo)
```

### Text Similarity Evaluator

**When to use**: Text where meaning matters more than exact wording.

**How it works**: Uses embeddings to measure semantic similarity. Compares the meaning of text, not just the words.

**Example**: 
```python
# Different wording, same meaning
extracted = "The camera quality is excellent"
expected = "Great photo capabilities"
# Returns ~0.85 (high semantic similarity despite different words)
```

### Score Judge

**When to use**: Numeric scores or ratings that need quality assessment.

**How it works**: Uses an LLM to evaluate the quality of extracted scores. The LLM judges whether the extracted score is reasonable given the context.

**Example**: 
```python
# Rating extraction
extracted = 4.5
expected = 5
# LLM evaluates: "Is 4.5 a reasonable rating given the review context?"
# Returns score based on LLM's judgment
```

### Label Model Grader

**When to use**: Classification labels that need context-aware evaluation.

**How it works**: Uses an LLM to evaluate if extracted labels are appropriate given the input context.

**Example**: 
```python
# Sentiment classification
extracted = "positive"
expected = "positive"
# LLM evaluates: "Is 'positive' the correct sentiment for this review?"
# Returns score based on LLM's judgment
```

### Python Code Evaluator

**When to use**: Custom evaluation logic that doesn't fit standard patterns.

**How it works**: Uses a callable you provide for evaluation.

**Example**: 
```python
def evaluate(extracted, expected, input_data=None, field_path=None):
    # Check if extracted value is within 10% of expected
    if abs(extracted - expected) / expected < 0.1:
        return 1.0
    return 0.0
```

### Predefined Score Evaluator

**When to use**: You already have scores and don't need to compute them.

**How it works**: Uses pre-computed scores directly. No evaluation needed—just use the scores you already have.

**Example**: 
```python
# Pre-computed scores from previous evaluation
examples_with_scores = [
    Example(..., predefined_score=0.95),
    Example(..., predefined_score=0.87),
]
```

## Per-Field Configuration

You can use different evaluators for different fields:

```python
evaluator_config = {
    "default": "exact",  # Most fields use exact match
    "field_overrides": {
        "name": "exact",  # Names must match exactly
        "description": "text_similarity",  # Descriptions use semantic similarity
        "rating": "score_judge",  # Ratings use LLM evaluation
    },
}
```

This allows fine-grained control over evaluation per field.

## Custom Evaluators

You can create custom evaluators by implementing the `BaseEvaluator` protocol:

```python
class MyEvaluator:
    def __init__(self, config: dict) -> None:
        # Initialize with config
        pass
    
    def evaluate(
        self,
        extracted: Any,
        expected: Any,
        input_data: dict | None = None,
        field_path: str | None = None,
    ) -> float:
        # Return score between 0.0 and 1.0
        return 0.85
```

## Evaluation During Optimization

During optimization, the prompter:

1. **Generate variation**: Create a new description/prompt variation
2. **Extract data**: Use the variation to extract from examples
3. **Evaluate**: Compare extracted data to expected data using evaluators
4. **Score**: Aggregate field scores into an overall score
5. **Select**: Choose variations with higher scores

## Choosing an Evaluator

Consider:

- **Data type**: Strings, numbers, lists, etc.
- **Matching requirements**: Exact vs. fuzzy vs. semantic
- **Performance needs**: Speed vs. accuracy
- **Domain requirements**: Business rules, constraints

## Trade-offs

### Exact Evaluator

- **Pros**: Fast, simple, unambiguous
- **Cons**: Too strict for text with variations

### Levenshtein Evaluator

- **Pros**: Handles minor variations, still fast
- **Cons**: May be too lenient, doesn't understand meaning

### Text Similarity Evaluator

- **Pros**: Understands meaning, handles paraphrasing
- **Cons**: Slower, requires embedding model

### LLM-based Evaluators

- **Pros**: Context-aware, handles complex cases
- **Cons**: Slower, more expensive, less deterministic

## Further Reading

- [Configure Evaluators](../guides/evaluators/configure.md)
- [Custom Evaluators](../guides/evaluators/custom.md)
- [When to Use Which](../guides/evaluators/selection.md)
- [Reference: Evaluators](../reference/api/evaluators.md)
- [Architecture](architecture.md)
