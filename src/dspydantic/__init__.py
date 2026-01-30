"""dspydantic - Optimize Pydantic model field descriptions using DSPy."""

# Import evaluators package to trigger registration - must be done before importing classes
import dspydantic.evaluators  # noqa: F401

# Now import the evaluator classes
from dspydantic.evaluators import (
    LabelModelGrader,
    LevenshteinEvaluator,
    PredefinedScoreEvaluator,
    PythonCodeEvaluator,
    ScoreJudge,
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
from dspydantic.persistence import PersistenceError
from dspydantic.prompter import ExtractionResult, Prompter
from dspydantic.types import Example, OptimizationResult, PrompterState, create_output_model
from dspydantic.utils import (
    image_to_base64,
    pdf_to_base64_images,
    prepare_input_data,
)

__version__ = "0.1.1"
__all__ = [
    "PydanticOptimizer",
    "Prompter",
    "ExtractionResult",
    "Example",
    "OptimizationResult",
    "PrompterState",
    "PersistenceError",
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
    "ScoreJudge",
    "LabelModelGrader",
    "PythonCodeEvaluator",
    "PredefinedScoreEvaluator",
]

