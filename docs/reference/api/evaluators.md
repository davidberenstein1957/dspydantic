# Evaluators

Evaluation system for measuring extraction quality.

## Evaluator Overview

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

## API Reference

::: dspydantic.evaluators.config.BaseEvaluator
    options:
      show_root_heading: true
      show_source: true

::: dspydantic.evaluators.StringCheckEvaluator
    options:
      show_root_heading: true
      show_source: true

::: dspydantic.evaluators.LevenshteinEvaluator
    options:
      show_root_heading: true
      show_source: true

::: dspydantic.evaluators.TextSimilarityEvaluator
    options:
      show_root_heading: true
      show_source: true

::: dspydantic.evaluators.ScoreJudge
    options:
      show_root_heading: true
      show_source: true

::: dspydantic.evaluators.LabelModelGrader
    options:
      show_root_heading: true
      show_source: true

::: dspydantic.evaluators.PythonCodeEvaluator
    options:
      show_root_heading: true
      show_source: true

::: dspydantic.evaluators.PredefinedScoreEvaluator
    options:
      show_root_heading: true
      show_source: true

## See Also

- [Configure Evaluators](../../guides/evaluators/configure.md)
- [When to Use Which](../../guides/evaluators/selection.md)
- [Custom Evaluators](../../guides/evaluators/custom.md)
- [Understanding Evaluators](../../concepts/evaluators.md)
