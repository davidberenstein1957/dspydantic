"""Integration tests for unified Prompter workflow."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy
from pydantic import BaseModel, Field

from dspydantic import Prompter
from dspydantic.evaluators import PredefinedScoreEvaluator
from dspydantic.types import Example, OptimizationResult


class Transaction(BaseModel):
    """Transaction model for testing."""

    broker: str = Field(description="Financial institution")
    amount: str = Field(description="Transaction amount")
    security: str = Field(description="Financial instrument")


def test_unified_workflow_optimize_save_load_predict():
    """Test complete workflow: optimize -> save -> load -> predict."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(model=Transaction)

    # Mock optimization
    with patch("dspydantic.prompter.PydanticOptimizer") as mock_optimizer_class:
        mock_optimizer = MagicMock()
        mock_result = OptimizationResult(
            optimized_descriptions={
                "broker": "The financial institution or brokerage firm",
                "amount": "The transaction amount with currency",
                "security": "The stock, bond, or financial instrument",
            },
            optimized_system_prompt="You are a financial document analysis assistant.",
            optimized_instruction_prompt="Extract transaction details from the financial report.",
            metrics={"average_score": 0.9, "baseline_score": 0.7},
            baseline_score=0.7,
            optimized_score=0.9,
        )
        mock_optimizer.optimize.return_value = mock_result
        mock_optimizer.model = Transaction
        mock_optimizer_class.return_value = mock_optimizer

        examples = [
            Example(
                text="Goldman Sachs processed a $2.5M equity trade for Tesla Inc.",
                expected_output={
                    "broker": "Goldman Sachs",
                    "amount": "$2.5M",
                    "security": "Tesla Inc.",
                },
            )
        ]

        result = prompter.optimize(examples=examples)

    # Verify optimization state
    assert prompter.optimized_descriptions
    assert prompter.optimized_system_prompt == "You are a financial document analysis assistant."

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_prompter"
        prompter.save(save_path)

        # Load (DSPy must be configured before loading)
        loaded = Prompter.load(save_path, model=Transaction)

        # Verify loaded state
        assert loaded.optimized_descriptions == prompter.optimized_descriptions
        assert loaded.optimized_system_prompt == prompter.optimized_system_prompt
        assert loaded.optimized_instruction_prompt == prompter.optimized_instruction_prompt

        # Mock prediction
        with patch("dspydantic.prompter.dspy") as mock_dspy:
            mock_lm = MagicMock()
            mock_dspy.settings.lm = mock_lm
            mock_dspy.LM.return_value = mock_lm
            mock_dspy.configure = MagicMock()

            mock_result = MagicMock()
            mock_result.json_output = '{"broker": "JPMorgan", "amount": "$500K", "security": "Apple Corp"}'
            mock_extractor = MagicMock()
            mock_extractor.return_value = mock_result
            mock_dspy.ChainOfThought.return_value = mock_extractor

            # Predict
            extracted = loaded.predict(text="JPMorgan executed $500K bond purchase for Apple Corp")

            assert extracted.broker == "JPMorgan"
            assert extracted.amount == "$500K"
            assert extracted.security == "Apple Corp"


def test_predefined_score_evaluator_with_prompter():
    """Test using PredefinedScoreEvaluator with Prompter."""
    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    prompter = Prompter(model=Transaction)

    # Pre-defined scores
    scores = [0.95, 0.87, 0.92]
    evaluator = PredefinedScoreEvaluator(config={"scores": scores})

    # Mock optimization
    with patch("dspydantic.prompter.PydanticOptimizer") as mock_optimizer_class:
        mock_optimizer = MagicMock()
        mock_result = OptimizationResult(
            optimized_descriptions={"broker": "Optimized"},
            optimized_system_prompt=None,
            optimized_instruction_prompt=None,
            metrics={},
            baseline_score=0.5,
            optimized_score=0.9,
        )
        mock_optimizer.optimize.return_value = mock_result
        mock_optimizer_class.return_value = mock_optimizer

        examples = [
            Example(text="Goldman Sachs $2.5M Tesla", expected_output={"broker": "Goldman Sachs"}),
            Example(text="JPMorgan $500K Apple", expected_output={"broker": "JPMorgan"}),
            Example(text="Morgan Stanley $1M Google", expected_output={"broker": "Morgan Stanley"}),
        ]

        # Optimize with predefined scores
        result = prompter.optimize(examples=examples, evaluate_fn=evaluator)

        # Verify optimizer was called with evaluator
        mock_optimizer_class.assert_called_once()
        # The evaluator should have been passed to the optimizer
        call_kwargs = mock_optimizer_class.call_args[1]
        assert call_kwargs["evaluate_fn"] == evaluator


def test_backward_compatibility_with_pydantic_optimizer():
    """Test backward compatibility - PydanticOptimizer still works."""
    from dspydantic import create_optimized_model

    # Simulate OptimizationResult from PydanticOptimizer
    result = OptimizationResult(
        optimized_descriptions={"broker": "Optimized broker description"},
        optimized_system_prompt=None,
        optimized_instruction_prompt=None,
        metrics={},
        baseline_score=0.5,
        optimized_score=0.9,
    )

    # Configure DSPy first
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key")
    dspy.configure(lm=lm)

    # Convert to Prompter
    prompter = Prompter.from_optimization_result(
        model=Transaction,
        optimization_result=result,
    )

    assert prompter.optimized_descriptions == {"broker": "Optimized broker description"}

    # Can also use create_optimized_model (old way)
    OptimizedModel = create_optimized_model(Transaction, result.optimized_descriptions)
    assert OptimizedModel is not None
    # Verify it's a different class but compatible
    assert OptimizedModel.__name__ != Transaction.__name__
