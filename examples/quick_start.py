from pydantic import BaseModel, Field
from typing import Literal
from dspydantic import PydanticOptimizer, Example, create_optimized_model

# 1. Define your model (any Pydantic model works)
class TransactionRecord(BaseModel):
    broker: str = Field(description="Financial institution or brokerage firm")
    amount: str = Field(description="Transaction amount with currency")
    security: str = Field(description="Stock, bond, or financial instrument")
    date: str = Field(description="Transaction date")
    transaction_type: Literal["equity", "bond", "option", "future", "forex"] = Field(
        description="Type of financial instrument"
    )

# 2. Provide examples (just input text + expected output)
examples = [
    Example(
        text="Transaction Report: Goldman Sachs processed a $2.5M equity trade for Tesla Inc. on March 15, 2024.",
        expected_output=TransactionRecord(
            broker="Goldman Sachs",
            amount="$2.5M",
            security="Tesla Inc.",
            date="March 15, 2024",
            transaction_type="equity"
        )
    ),
    Example(
        text="JPMorgan executed $500K bond purchase for Apple Corp dated 2024-03-20.",
        expected_output=TransactionRecord(
            broker="JPMorgan",
            amount="$500K",
            security="Apple Corp",
            date="2024-03-20",
            transaction_type="bond"
        )
    ),
]

# 3. Optimize and use
optimizer = PydanticOptimizer(
    model=TransactionRecord,
    examples=examples,
    model_id="gemini/gemini-2.5-flash-lite",
    system_prompt="You are a financial document analysis assistant.",
    instruction_prompt="Extract transaction details from the financial report.",
)
result = optimizer.optimize() 

OptimizedTransactionRecord = create_optimized_model(
    TransactionRecord,
    result.optimized_descriptions
)
print(result.optimized_descriptions)
print(result.optimized_system_prompt)
print(result.optimized_instruction_prompt)
# Use OptimizedTransactionRecord just like your original model, but with better accuracy!