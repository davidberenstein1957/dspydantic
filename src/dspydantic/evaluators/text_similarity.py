"""Text similarity evaluator using embeddings."""

from typing import Any


class TextSimilarityEvaluator:
    """Evaluator that uses embeddings for semantic similarity.

    Best for text where meaning matters more than exact wording. Uses embedding
    models to compute cosine similarity between extracted and expected values.

    Args:
        config: Configuration dictionary with options:
            - model (str): Embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")
            - provider (str): "sentence-transformers" or "openai" (default: "sentence-transformers")
            - api_key (str): API key for OpenAI provider
            - threshold (float): Minimum similarity (0-1, default: 0.0)

    Raises:
        ImportError: If sentence-transformers is not installed when using that provider.

    Example:
        >>> # Requires: pip install sentence-transformers
        >>> evaluator = TextSimilarityEvaluator(config={})  # doctest: +SKIP
        >>> evaluator.evaluate("CEO", "Chief Executive Officer")  # doctest: +SKIP
        0.82  # Semantically similar

        With OpenAI embeddings:

        >>> evaluator = TextSimilarityEvaluator(config={  # doctest: +SKIP
        ...     "provider": "openai",
        ...     "model": "text-embedding-ada-002",
        ...     "api_key": "your-key"
        ... })
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize TextSimilarityEvaluator.

        Args:
            config: Configuration dictionary with model, provider, api_key, threshold options.
        """
        self.config = config
        self.model_name = config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.provider = config.get("provider", "sentence-transformers")
        self.api_key = config.get("api_key")
        self.threshold = config.get("threshold", 0.0)
        self._embedder = None

    def _get_embedder(self):
        """Lazy load the embedding model."""
        if self._embedder is not None:
            return self._embedder

        if self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for TextSimilarityEvaluator. "
                    "Install it with: pip install sentence-transformers"
                )
        elif self.provider == "openai":
            self._embedder = ("openai", self.model_name, self.api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._embedder

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for texts."""
        embedder = self._get_embedder()

        if self.provider == "sentence-transformers":
            return embedder.encode(texts, convert_to_numpy=False).tolist()
        elif self.provider == "openai":
            import openai

            _, model_name, api_key = embedder
            client = openai.OpenAI(api_key=api_key)
            response = client.embeddings.create(model=model_name, input=texts)
            return [item.embedding for item in response.data]

        raise ValueError(f"Unknown provider: {self.provider}")

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(a * a for a in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def evaluate(
        self,
        extracted: Any,
        expected: Any,
        input_data: dict[str, Any] | None = None,
        field_path: str | None = None,
    ) -> float:
        """Evaluate using semantic similarity via embeddings.

        Args:
            extracted: Extracted value.
            expected: Expected value.
            input_data: Optional input data (not used).
            field_path: Optional field path (not used).

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        extracted_str = str(extracted)
        expected_str = str(expected)

        if extracted_str == expected_str:
            return 1.0

        try:
            embeddings = self._get_embeddings([extracted_str, expected_str])
            similarity = self._cosine_similarity(embeddings[0], embeddings[1])
            return max(0.0, similarity) if similarity >= self.threshold else 0.0
        except Exception:
            # Fallback to exact match if embeddings fail
            return 1.0 if extracted_str == expected_str else 0.0
