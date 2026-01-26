"""dspydantic - Optimize Pydantic model field descriptions using DSPy."""

# Import evaluators package to trigger registration - must be done before importing classes
import dspydantic.evaluators  # noqa: F401

# Now import the evaluator classes
from dspydantic.evaluators import (
    LabelModelGrader,
    LevenshteinEvaluator,
    PythonCodeEvaluator,
    ScoreModelGrader,
    StringCheckEvaluator,
    TextSimilarityEvaluator,
)
from dspydantic.evaluators.config import (
    EVALUATOR_REGISTRY,
    BaseEvaluator,
    EvaluatorFactory,
    register_evaluator,
)
from dspydantic.extractor import (
    apply_optimized_descriptions,
    create_optimized_model,
    extract_field_descriptions,
)
from dspydantic.optimizer import PydanticOptimizer
from dspydantic.types import Example, OptimizationResult, create_output_model
from dspydantic.utils import (
    image_to_base64,
    pdf_to_base64_images,
    prepare_input_data,
)

__version__ = "0.0.7"
__all__ = [
    "PydanticOptimizer",
    "Example",
    "OptimizationResult",
    "extract_field_descriptions",
    "apply_optimized_descriptions",
    "create_optimized_model",
    "prepare_input_data",
    "image_to_base64",
    "pdf_to_base64_images",
    "create_output_model",
    # Evaluator system
    "BaseEvaluator",
    "EvaluatorFactory",
    "EVALUATOR_REGISTRY",
    "register_evaluator",
    "StringCheckEvaluator",
    "LevenshteinEvaluator",
    "TextSimilarityEvaluator",
    "ScoreModelGrader",
    "LabelModelGrader",
    "PythonCodeEvaluator",
]

