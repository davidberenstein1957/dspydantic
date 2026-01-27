import json
import os
from typing import Any

from app.models import LabeledExample, PromptVersion, Task
from app.services.schema_service import create_pydantic_model_from_schema


def convert_examples_to_dspydantic_format(examples: list[LabeledExample]) -> list[Any]:
    """Convert labeled examples to dspydantic Example format."""
    try:
        from dspydantic import Example
    except ImportError:
        raise ImportError("dspydantic is not installed. Install it with: pip install dspydantic")

    dspydantic_examples = []
    for example in examples:
        input_data = example.input_data or {}
        expected_output = example.output_data

        # Build Example kwargs
        example_kwargs = {"expected_output": expected_output}

        # Handle text (string or dict for template prompts)
        if "text" in input_data:
            example_kwargs["text"] = input_data["text"]

        # Handle images (list of base64 strings or paths)
        if "images" in input_data and input_data["images"]:
            images = input_data["images"]
            if isinstance(images, list) and len(images) > 0:
                # Use first image - dspydantic Example supports one image at a time
                image_data = images[0]
                if isinstance(image_data, str):
                    if image_data.startswith("data:image"):
                        example_kwargs["image_base64"] = image_data
                    elif image_data.startswith("http"):
                        # URL - would need to download, for now treat as path
                        example_kwargs["image_path"] = image_data
                    else:
                        # Assume it's a file path
                        example_kwargs["image_path"] = image_data

        # Handle PDF
        if "pdf" in input_data and input_data["pdf"]:
            pdf_data = input_data["pdf"]
            if isinstance(pdf_data, str):
                example_kwargs["pdf_path"] = pdf_data

        # If no specific input type found, use input_data as text (dict or string)
        if "text" not in example_kwargs and "image_path" not in example_kwargs and "image_base64" not in example_kwargs and "pdf_path" not in example_kwargs:
            example_kwargs["text"] = input_data if input_data else ""

        dspydantic_examples.append(Example(**example_kwargs))

    return dspydantic_examples


def run_optimization(
    task: Task,
    examples: list[LabeledExample],
    config: dict[str, Any],
    prompt_version: PromptVersion | None = None
) -> dict[str, Any]:
    """
    Run DSPy optimization on a task with examples using dspydantic.
    Returns optimization results including metrics and optimized descriptions.
    """
    try:
        import dspy

        from dspydantic import Example, PydanticOptimizer

        # Create Pydantic model from schema
        model = create_pydantic_model_from_schema(task.pydantic_schema)

        # Convert examples to dspydantic format
        dspydantic_examples = convert_examples_to_dspydantic_format(examples)

        # Get configuration
        # Use config model_id if provided, otherwise task default_model, otherwise env var
        model_id = config.get("model_id") or task.default_model or os.getenv("DSPY_MODEL_ID", "gpt-4o")
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")

        # Metric maps to evaluate_fn
        metric = config.get("metric", "exact")
        evaluate_fn = metric  # "exact", "levenshtein"

        # Optimizer (None means auto-select)
        optimizer_type = config.get("optimizer")  # None for auto, or specific optimizer name

        # Train split: convert percentage (0-100) to fraction (0.0-1.0)
        train_split_percent = config.get("train_split", 80.0)
        train_split = max(0.0, min(1.0, train_split_percent / 100.0))

        # Initialize DSPy LM if needed
        if api_key:
            dspy.configure(lm=dspy.LM(model=model_id, api_key=api_key))

        # Get prompts from prompt version if provided, otherwise use task defaults
        if prompt_version:
            system_prompt = prompt_version.system_prompt
            instruction_prompt = prompt_version.instruction_prompt
        else:
            system_prompt = task.system_prompt
            instruction_prompt = task.instruction_prompt_template

        # Create optimizer with all parameters
        optimizer_kwargs = {
            "model": model,
            "examples": dspydantic_examples,
            "evaluate_fn": evaluate_fn,
            "model_id": model_id,
            "train_split": train_split,  # Fraction (0.0-1.0) for train/test split
        }

        # Add optional parameters
        if api_key:
            optimizer_kwargs["api_key"] = api_key
        if optimizer_type:
            optimizer_kwargs["optimizer"] = optimizer_type
        if system_prompt:
            optimizer_kwargs["system_prompt"] = system_prompt
        if instruction_prompt:
            optimizer_kwargs["instruction_prompt"] = instruction_prompt

        optimizer = PydanticOptimizer(**optimizer_kwargs)

        # Run optimization
        result = optimizer.optimize()

        # Format results
        results = {
            "optimized_prompt": json.dumps(result.optimized_descriptions, indent=2),
            "optimized_descriptions": result.optimized_descriptions,
            "optimized_system_prompt": getattr(result, "optimized_system_prompt", None),
            "optimized_instruction_prompt": getattr(result, "optimized_instruction_prompt", None),
            "metrics": getattr(result, "metrics", {}),
            "config": config,
        }

        return results
    except ImportError as e:
        raise ImportError(f"Required dependencies not installed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Optimization failed: {str(e)}")


def run_evaluation(
    task: Task,
    examples: list[LabeledExample],
    config: dict[str, Any],
    prompt_version: PromptVersion | None = None
) -> dict[str, Any]:
    """
    Run evaluation on a task with examples using dspydantic.
    Evaluates the model against the dataset using the specified prompts.
    Returns evaluation results including metrics.
    """
    try:
        import dspy

        from dspydantic import Example
        from dspydantic.evaluators.functions import default_evaluate_fn
        from dspydantic.extractor import extract_field_descriptions

        # Create Pydantic model from schema
        model = create_pydantic_model_from_schema(task.pydantic_schema)

        # Convert examples to dspydantic format
        dspydantic_examples = convert_examples_to_dspydantic_format(examples)

        # Get configuration
        model_id = config.get("model_id") or task.default_model or os.getenv("DSPY_MODEL_ID", "gpt-4o")
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        metric = config.get("metric", "exact")

        # Initialize DSPy LM
        if api_key:
            lm = dspy.LM(model=model_id, api_key=api_key)
        else:
            lm = dspy.LM(model=model_id)

        # Use context manager for thread-local configuration
        # This works in background threads without trying to change global settings
        # All dspy operations within this context will use the provided LM
        with dspy.settings.context(lm=lm):
            # Get prompts and field descriptions
            # Use prompt version if provided, otherwise use task defaults
            if prompt_version:
                system_prompt = prompt_version.system_prompt
                instruction_prompt = prompt_version.instruction_prompt
                field_descriptions = prompt_version.output_schema_descriptions or {}
            else:
                system_prompt = task.system_prompt
                instruction_prompt = task.instruction_prompt_template
                field_descriptions = {}

            # If no field descriptions from prompt version, extract from schema
            if not field_descriptions:
                field_descriptions = extract_field_descriptions(model)

            # Create evaluation function
            evaluate_fn = default_evaluate_fn(
                lm=lm,
                model=model,
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                metric=metric
            )

            # Import for comparison
            from deepdiff import DeepDiff

            # Evaluate all examples and capture detailed results
            scores = []
            example_results = []
            total_examples = len(dspydantic_examples)

            for i, (example, labeled_example) in enumerate(zip(dspydantic_examples, examples)):
                try:
                    # Get expected output
                    expected_output = labeled_example.output_data

                    # Extract data using the model
                    from dspydantic.extractor import apply_optimized_descriptions
                    modified_schema = apply_optimized_descriptions(model, field_descriptions)

                    # Build extraction prompt
                    system_prompt_to_use = system_prompt or ""
                    instruction_prompt_raw = instruction_prompt or ""
                    from dspydantic.utils import format_instruction_prompt_template
                    instruction_prompt_to_use = format_instruction_prompt_template(instruction_prompt_raw, example.text_dict) or ""

                    # Extract using LM
                    prompt_parts = []
                    if system_prompt_to_use:
                        prompt_parts.append(f"System: {system_prompt_to_use}")
                    if instruction_prompt_to_use:
                        prompt_parts.append(f"Instruction: {instruction_prompt_to_use}")
                    prompt_parts.append(f"\nJSON Schema:\n{json.dumps(modified_schema, indent=2)}")

                    input_data = example.input_data
                    if isinstance(input_data, dict):
                        if "text" in input_data:
                            prompt_parts.append(f"\nInput text: {input_data['text']}")
                        if "images" in input_data:
                            prompt_parts.append(f"\nInput images: {len(input_data['images'])} image(s) provided")
                    else:
                        prompt_parts.append(f"\nInput: {str(input_data)}")

                    prompt_parts.append("\nExtract the structured data according to the JSON schema above and return it as valid JSON.")
                    full_prompt = "\n\n".join(prompt_parts)
                    json_prompt = f"{full_prompt}\n\nReturn only valid JSON, no other text."

                    # Use DSPy to extract
                    import dspy

                    from dspydantic.utils import (
                        build_image_signature_and_kwargs,
                        convert_images_to_dspy_images,
                    )

                    dspy_images = None
                    if isinstance(input_data, dict) and "images" in input_data and input_data["images"]:
                        try:
                            dspy_images = convert_images_to_dspy_images(input_data["images"])
                        except:
                            pass

                    signature, extractor_kwargs = build_image_signature_and_kwargs(dspy_images)
                    extractor = dspy.ChainOfThought(signature)
                    extractor_kwargs["prompt"] = json_prompt
                    result = extractor(**extractor_kwargs)

                    # Parse extracted output
                    output_text = str(result.json_output) if hasattr(result, "json_output") else str(result)
                    extracted_output = None
                    try:
                        extracted_output = json.loads(output_text)
                    except:
                        import re
                        json_match = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", output_text, re.DOTALL)
                        if json_match:
                            try:
                                extracted_output = json.loads(json_match.group())
                            except:
                                pass

                    # Calculate score and differences
                    score = evaluate_fn(example, field_descriptions, system_prompt, instruction_prompt)

                    # Calculate differences
                    differences = None
                    if extracted_output is not None and expected_output is not None:
                        try:
                            diff = DeepDiff(expected_output, extracted_output, ignore_order=False, verbose_level=1)
                            if diff:
                                differences = {
                                    "type_changes": [str(v) for v in diff.get("type_changes", {}).values()],
                                    "values_changed": {k: {"old": str(v.get("old_value", "")), "new": str(v.get("new_value", ""))} for k, v in diff.get("values_changed", {}).items()},
                                    "dictionary_item_added": list(diff.get("dictionary_item_added", [])),
                                    "dictionary_item_removed": list(diff.get("dictionary_item_removed", [])),
                                    "iterable_item_added": list(diff.get("iterable_item_added", [])),
                                    "iterable_item_removed": list(diff.get("iterable_item_removed", [])),
                                }
                        except:
                            differences = {"error": "Could not calculate differences"}

                    example_results.append({
                        "example_id": labeled_example.id,
                        "score": score,
                        "extracted_output": extracted_output,
                        "expected_output": expected_output,
                        "differences": differences,
                        "error_message": None
                    })
                    scores.append(score)
                except Exception as e:
                    error_msg = str(e)
                    example_results.append({
                        "example_id": labeled_example.id,
                        "score": 0.0,
                        "extracted_output": None,
                        "expected_output": labeled_example.output_data,
                        "differences": None,
                        "error_message": error_msg
                    })
                    scores.append(0.0)

            # Calculate metrics
            avg_score = sum(scores) / len(scores) if scores else 0.0
            exact_matches = sum(1 for s in scores if s == 1.0)
            exact_match_rate = exact_matches / total_examples if total_examples > 0 else 0.0

            metrics = {
                "average_score": avg_score,
                "exact_match_rate": exact_match_rate,
                "exact_matches": exact_matches,
                "total_examples": total_examples,
                "scores": scores,
                "metric": metric
            }

            results = {
                "metrics": metrics,
                "config": config,
                "example_results": example_results,
            }

            return results
    except ImportError as e:
        raise ImportError(f"Required dependencies not installed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Evaluation failed: {str(e)}")
