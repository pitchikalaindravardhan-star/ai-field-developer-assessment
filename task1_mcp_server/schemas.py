"""Input models for the two tools.

These models are the single source of truth: the JSON Schema advertised in
tools/list is generated from them, so what clients are told and what the server
enforces cannot drift apart.
"""

from decimal import Decimal
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

# [0-9] rather than \d: pydantic's regex engine treats \d as any Unicode digit,
# which would let Arabic-Indic numerals through.
CustomerId = Annotated[str, StringConstraints(pattern=r"^CUST-[0-9]{5}$")]


def _at_most_two_decimals(value: float) -> float:
    # repr-based on purpose: Decimal(str(19.99)) is 19.99, Decimal(19.99) is not.
    if Decimal(str(value)).as_tuple().exponent < -2:
        raise ValueError("amount can have at most 2 decimal places")
    return value


Amount = Annotated[
    float,
    Field(gt=0, allow_inf_nan=False, description="Refund amount, must be above zero"),
    AfterValidator(_at_most_two_decimals),
]

Reason = Annotated[
    str,
    # Stripped first, so ten spaces do not count as a reason.
    StringConstraints(strip_whitespace=True, min_length=10, max_length=500),
]


class ToolInput(BaseModel):
    # strict stops "100" -> 100.0 and True -> 1.0; extra=forbid catches typos
    # such as customerId that would otherwise be dropped without complaint.
    model_config = ConfigDict(strict=True, extra="forbid")


class GetCustomerRecordInput(ToolInput):
    customer_id: CustomerId = Field(description="Customer id, for example CUST-10001")


class TriggerRefundInput(ToolInput):
    customer_id: CustomerId = Field(description="Customer id, for example CUST-10001")
    amount: Amount
    reason: Reason = Field(description="Why the refund is issued, at least 10 characters")
