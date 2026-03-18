# Types

Core types and data structures.

::: dspydantic.types.Example
    options:
      show_root_heading: true
      show_source: true

::: dspydantic.types.OptimizationResult
    options:
      show_root_heading: true
      show_source: true

::: dspydantic.prompter.ExtractionResult
    options:
      show_root_heading: true
      show_source: true

::: dspydantic.types.PrompterState
    options:
      show_root_heading: true
      show_source: true

::: dspydantic.types.FieldOptimizationProgress
    options:
      show_root_heading: true
      show_source: true

## Example

The `Example` class represents a single example for optimization. It supports multiple input types:

- **Text**: Plain text string or dictionary for prompt templates
- **Images**: File path (`image_path`) or base64-encoded string (`image_base64`)
- **PDFs**: File path (`pdf_path`) - automatically converted to images at specified DPI (default: 300)

PDFs are converted to images page by page for processing. Use `pdf_dpi` parameter to control conversion quality (default: 300 DPI).

## OptimizationResult

The `OptimizationResult` dataclass contains the results of optimization:

- `optimized_descriptions`: Dictionary mapping field paths to optimized descriptions
- `optimized_system_prompt`: Optimized system prompt (if provided)
- `optimized_instruction_prompt`: Optimized instruction prompt (if provided)
- `metrics`: Dictionary containing optimization metrics
- `baseline_score`: Baseline score before optimization
- `optimized_score`: Score after optimization
- `api_calls`: Total API calls made during optimization
- `total_tokens`: Total tokens used during optimization

## ExtractionResult

The `ExtractionResult` dataclass is returned by `predict_with_confidence()`:

- `data`: The extracted Pydantic model instance
- `confidence`: Confidence score (0.0-1.0)
- `raw_output`: Raw LLM output text (optional)

## PrompterState

The `PrompterState` dataclass contains all information needed to save and restore a Prompter instance.

## FieldOptimizationProgress

The `FieldOptimizationProgress` dataclass is emitted by the `on_progress` callback during optimization to track progress:

- `phase`: Current optimization phase ("baseline", "fields", "skipped", "system_prompt", "instruction_prompt", "complete")
- `score_before`: Score before this optimization step
- `score_after`: Score after this optimization step
- `improved`: Whether the score improved
- `total_fields`: Total number of fields being optimized
- `field_path`: Dot-notation path of the field being optimized (None for non-field phases)
- `field_index`: 1-based index of the field (None for non-field phases)
- `elapsed_seconds`: Wall-clock seconds elapsed since optimization started
- `optimized_value`: The actual optimized description or prompt text (new in v0.1.3+)

### Usage with Callbacks

```python
def my_progress_callback(progress: FieldOptimizationProgress):
    if progress.phase == "fields":
        print(f"{progress.field_path}: {progress.score_before:.0%} → {progress.score_after:.0%}")
        if progress.optimized_value:
            print(f"  Optimized to: {progress.optimized_value!r}")

optimizer.on_progress = my_progress_callback
```

Callbacks are automatically invoked when `verbose=True` with rich-formatted output showing optimized values.

## See Also

- [Optimization Modalities](../../guides/optimization/modalities.md)
- [Optimize with Templates](../../guides/optimization/prompt-templates.md)
- [Save and Load Prompters](../../guides/advanced/save-load.md)
