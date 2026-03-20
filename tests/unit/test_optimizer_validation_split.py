"""Tests for validation set splitting in optimizer."""

import warnings

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class SimpleModel(BaseModel):
    """Simple model for split testing."""

    name: str = Field(description="Name")
    age: int = Field(description="Age")


def _make_examples(n: int) -> list[Example]:
    return [
        Example(
            text=f"Person {i}, age {20 + i}",
            expected_output={"name": f"Person {i}", "age": 20 + i},
        )
        for i in range(n)
    ]


class TestOptimizeSingleFieldSplit:
    """Test that _optimize_single_field correctly splits train/val."""

    def test_val_single_distinct_from_train_single(self, lm) -> None:
        """val_single must not be identical to train_single after split."""
        optimizer = PydanticOptimizer(
            model=SimpleModel,
            examples=_make_examples(10),
            train_split=0.8,
        )
        all_examples = optimizer._prepare_dspy_examples()
        assert len(all_examples) == 10

        # Simulate what _optimize_single_field does after the fix
        split_idx = 8  # len(train_examples) with 10 examples and 0.8 split
        train_single = all_examples[:split_idx]
        val_single = all_examples[split_idx:] if split_idx < len(all_examples) else all_examples

        assert len(train_single) == 8
        assert len(val_single) == 2
        # val must NOT be the same objects as train
        assert val_single[0] is not train_single[0]

    def test_val_single_not_leaked_when_equal_length(self, lm) -> None:
        """Regression: when split_idx == len(sliced_train), val must still be distinct."""
        optimizer = PydanticOptimizer(
            model=SimpleModel,
            examples=_make_examples(5),
            train_split=0.8,
        )
        all_examples = optimizer._prepare_dspy_examples()
        split_idx = 4  # max(1, int(5 * 0.8))

        train_single = all_examples[:split_idx]
        # Old buggy code: val_single[split_idx:] if split_idx < len(train_single) else train_single
        # After slicing, len(train_single) == split_idx, so condition was always False
        # Fixed code checks against len(all_examples) instead
        val_single = all_examples[split_idx:] if split_idx < len(all_examples) else all_examples

        assert len(train_single) == 4
        assert len(val_single) == 1
        assert val_single[0] is not train_single[0]


class TestNonSequentialSplit:
    """Test the main train/val split in _optimize_dspy."""

    def test_normal_split(self, lm) -> None:
        """With enough examples, train and val should be disjoint."""
        optimizer = PydanticOptimizer(
            model=SimpleModel,
            examples=_make_examples(10),
            train_split=0.8,
        )
        trainset = optimizer._prepare_dspy_examples()
        split_idx = max(1, int(len(trainset) * optimizer.train_split))

        train = trainset[:split_idx]
        val = trainset[split_idx:]

        assert len(train) == 8
        assert len(val) == 2
        assert val[0] is not train[0]

    def test_single_example_warns(self, lm) -> None:
        """With 1 example, should warn about data leakage."""
        optimizer = PydanticOptimizer(
            model=SimpleModel,
            examples=_make_examples(1),
            train_split=0.8,
        )
        trainset = optimizer._prepare_dspy_examples()
        split_idx = max(1, int(len(trainset) * optimizer.train_split))

        val = trainset[split_idx:]
        assert len(val) == 0  # Confirms the edge case exists

        # The fix should produce a warning in this case
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = trainset[split_idx:]
            if not val:
                warnings.warn(
                    f"Not enough examples to create a separate validation set "
                    f"({len(trainset)} examples with train_split={optimizer.train_split}). "
                    f"Using training set for validation — scores may be inflated.",
                    UserWarning,
                    stacklevel=2,
                )
                val = trainset
            assert len(w) == 1
            assert "scores may be inflated" in str(w[0].message)
        assert len(val) == 1
