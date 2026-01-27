"""Evaluator implementations for dspydantic."""

from dspydantic.evaluators.config import register_evaluator
from dspydantic.evaluators.label_model_grader import LabelModelGrader
from dspydantic.evaluators.levenshtein import LevenshteinEvaluator
from dspydantic.evaluators.predefined_score import PredefinedScoreEvaluator
from dspydantic.evaluators.python_code import PythonCodeEvaluator
from dspydantic.evaluators.score_judge import ScoreJudge
from dspydantic.evaluators.string_check import StringCheckEvaluator
from dspydantic.evaluators.text_similarity import TextSimilarityEvaluator

# Register built-in evaluators
register_evaluator("exact", StringCheckEvaluator)
register_evaluator("levenshtein", LevenshteinEvaluator)
register_evaluator("string_check", StringCheckEvaluator)
register_evaluator("text_similarity", TextSimilarityEvaluator)
register_evaluator("score_judge", ScoreJudge)
register_evaluator("score_model_grader", ScoreJudge)  # backward compat
register_evaluator("label_model_grader", LabelModelGrader)
register_evaluator("python_code", PythonCodeEvaluator)
register_evaluator("predefined_score", PredefinedScoreEvaluator)

__all__ = [
    "StringCheckEvaluator",
    "LevenshteinEvaluator",
    "TextSimilarityEvaluator",
    "ScoreJudge",
    "LabelModelGrader",
    "PythonCodeEvaluator",
    "PredefinedScoreEvaluator",
]
