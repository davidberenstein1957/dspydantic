"""Static analysis of optimization approaches - shows compile counts and estimated costs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pydantic import BaseModel, Field
from dspydantic import Example


class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Product price in USD")
    category: str = Field(description="Product category")


def analyze_optimization():
    """Analyze different optimization strategies."""

    # Count fields and examples
    model = Product
    num_fields = len(model.model_fields)
    num_examples = 6  # From benchmark_instrumented.py

    print("="*80)
    print("DSPy OPTIMIZATION STRATEGY ANALYSIS")
    print("="*80)
    print(f"\nTest Setup:")
    print(f"  Fields: {num_fields} (name, price, category)")
    print(f"  Examples: {num_examples}")
    print(f"  Optimizer: BootstrapFewShot (default)")
    print()

    # Strategy 1: Single-pass
    print("STRATEGY 1: Single-Pass (Default, fast=False)")
    print("-" * 80)
    single_pass_compiles = 1
    single_pass_calls = 50  # Baseline + 1 compile
    print(f"  What happens:")
    print(f"    1. Run 1 compile with all fields + system + instruction prompts")
    print(f"    → Treats all 3 fields as one optimization problem")
    print(f"")
    print(f"  Compile count: {single_pass_compiles}")
    print(f"  API calls: ~{single_pass_calls} (baseline check + 1 compile)")
    print(f"  Time estimate: ~30s (1 compile × 25-35s per compile)")
    print(f"  Quality: Good (fields optimized together, may miss field-specific issues)")
    print()

    # Strategy 2: Sequential
    print("STRATEGY 2: Sequential Field-by-Field (sequential=True, parallel_fields=False)")
    print("-" * 80)
    sequential_compiles = num_fields + 2  # Each field + 2 prompts
    sequential_calls = sequential_compiles * 50
    print(f"  What happens:")
    print(f"    1. Baseline compile")
    print(f"    2. Compile field 1 (name)")
    print(f"    3. Compile field 2 (price)  ← Uses optimized field 1")
    print(f"    4. Compile field 3 (category) ← Uses optimized fields 1-2")
    print(f"    5. Compile system_prompt")
    print(f"    6. Compile instruction_prompt")
    print(f"  → Each field builds on previous improvements")
    print(f"")
    print(f"  Compile count: {sequential_compiles}")
    print(f"  API calls: ~{sequential_calls} ({sequential_compiles} compiles × 50 calls each)")
    print(f"  Time estimate: ~{sequential_compiles * 25}s ({sequential_compiles} compiles × 25-35s each)")
    print(f"  Quality: Excellent (per-field optimization, cumulative improvements)")
    print()

    # Strategy 3: Sequential + Parallel
    print("STRATEGY 3: Sequential + Parallel Fields (sequential=True, parallel_fields=True)")
    print("-" * 80)
    seq_parallel_compiles = 5  # All fields parallel, then 2 prompts
    seq_parallel_calls = seq_parallel_compiles * 50
    print(f"  What happens:")
    print(f"    1. Baseline compile")
    print(f"    2. Compile [field 1, field 2, field 3] IN PARALLEL (uses ThreadPoolExecutor)")
    print(f"    3. Compile system_prompt")
    print(f"    4. Compile instruction_prompt")
    print(f"")
    print(f"  Compile count: {seq_parallel_compiles}")
    print(f"  API calls: ~{seq_parallel_calls} ({seq_parallel_compiles} compiles × 50 calls each)")
    print(f"  Time estimate: ~{seq_parallel_compiles * 25}s (effectively 2-3 × 25s due to parallelism)")
    print(f"  Quality: Very Good (fields optimized independently, no dependencies)")
    print()

    # Speedup comparison
    print("\n" + "="*80)
    print("SPEEDUP ANALYSIS")
    print("="*80)
    print()

    single_pass_time = 25
    sequential_time = sequential_compiles * 25
    seq_parallel_time = 75  # 3 parallel rounds ~25s each

    print(f"Single-pass:           {single_pass_time:>3}s (baseline)")
    print(f"Sequential:            {sequential_time:>3}s ({sequential_time/single_pass_time:.1f}× slower)")
    print(f"Sequential + Parallel: {seq_parallel_time:>3}s ({seq_parallel_time/single_pass_time:.1f}× slower)")
    print()

    speedup_parallel = sequential_time / seq_parallel_time
    print(f"Parallelism benefit: {speedup_parallel:.1f}× speedup on sequential")
    print(f"  → Saves {sequential_time - seq_parallel_time}s by running field compiles in parallel")
    print()

    # Fast mode impact
    print("\n" + "="*80)
    print("FAST MODE IMPACT (max_bootstrapped_demos=1)")
    print("="*80)
    print()
    print("Fast mode reduces demo budget for BootstrapFewShot:")
    print("  Standard: max_bootstrapped_demos=16 (or 2)")
    print("  Fast:     max_bootstrapped_demos=1")
    print()
    print("Effect: ~30-40% faster per compile, slightly lower quality")
    print()
    print(f"With fast=True:")
    print(f"  Single-pass:           {int(single_pass_time * 0.65):>2}s")
    print(f"  Sequential:            {int(sequential_time * 0.65):>2}s")
    print(f"  Sequential + Parallel: {int(seq_parallel_time * 0.65):>2}s")
    print()

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    print("1. DEFAULT STRATEGY: Sequential + Parallel (no fast toggle)")
    print("   → Best quality/speed tradeoff for most users")
    print(f"   → {seq_parallel_time}s for excellent quality")
    print()
    print("2. FOR SPEED: Add max_val_examples parameter")
    print("   → Reduces evaluation set size (currently ~6 examples)")
    print("   → 3 examples = ~2× faster evaluation")
    print()
    print("3. KEEP API COST LOW:")
    print("   → Use num_threads=1 (sequential only, no parallel)")
    print("   → Use max_val_examples=2")
    print("   → Use optimizer_kwargs to set max_bootstrapped_demos=1")
    print()
    print("4. REMOVE: The 'fast' toggle")
    print("   → Instead: Replace with granular parameters")
    print("   → Let users pick: demo_budget, max_val_examples, parallel=True/False")


if __name__ == "__main__":
    analyze_optimization()
