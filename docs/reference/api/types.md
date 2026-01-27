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

::: dspydantic.types.PrompterState
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

## PrompterState

The `PrompterState` dataclass contains all information needed to save and restore a Prompter instance.

## See Also

- [Optimization Modalities](../../guides/optimization/modalities.md)
- [Optimize with Templates](../../guides/optimization/prompt-templates.md)
- [Save and Load Prompters](../../guides/advanced/save-load.md)
