"""Minimal test to verify Bedrock integration setup.

This script verifies that:
1. boto3 is available
2. The bedrock_example.py can be imported
3. Basic structure is correct

Does NOT require AWS credentials to run.
"""

import sys


def test_boto3_available():
    """Test that boto3 is available."""
    try:
        import boto3

        print("✓ boto3 is installed")
        return True
    except ImportError:
        print("✗ boto3 is NOT installed. Run: pip install boto3")
        return False


def test_bedrock_example_imports():
    """Test that bedrock_example can be imported."""
    try:
        # Just check syntax by trying to compile it
        with open("examples/bedrock_example.py") as f:
            code = f.read()
        compile(code, "bedrock_example.py", "exec")
        print("✓ bedrock_example.py syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ bedrock_example.py has syntax error: {e}")
        return False
    except FileNotFoundError:
        print("✗ bedrock_example.py not found")
        return False


def test_dspy_supports_bedrock():
    """Test that DSPy has Bedrock support."""
    try:
        import dspy

        # Check if LM class exists
        assert hasattr(dspy, "LM"), "dspy.LM not found"
        print("✓ DSPy LM class is available")

        # Note: We can't actually test Bedrock connection without credentials
        # but we can verify the module structure
        return True
    except ImportError:
        print("✗ dspy is NOT installed")
        return False
    except AssertionError as e:
        print(f"✗ DSPy check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Bedrock Integration Verification")
    print("=" * 60)
    print()

    results = []

    print("Testing dependencies...")
    results.append(test_boto3_available())
    results.append(test_dspy_supports_bedrock())
    print()

    print("Testing example files...")
    results.append(test_bedrock_example_imports())
    print()

    print("=" * 60)
    if all(results):
        print("✓ All checks passed!")
        print()
        print("To test with actual AWS Bedrock:")
        print("1. Configure AWS credentials (AWS_PROFILE or access keys)")
        print("2. Run: python examples/bedrock_example.py")
    else:
        print("✗ Some checks failed")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
