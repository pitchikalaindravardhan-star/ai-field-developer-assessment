"""In-memory stand-in for a customer database."""

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from uuid import uuid4


class CustomerNotFound(Exception):
    pass


class RefundExceedsBalance(Exception):
    pass


@dataclass
class Customer:
    customer_id: str
    name: str
    tier: str
    refundable_balance: Decimal
    refunds: list[dict] = field(default_factory=list)

    def public_view(self) -> dict:
        record = asdict(self)
        record["refundable_balance"] = float(self.refundable_balance)
        return record


class CustomerStore:
    def __init__(self) -> None:
        self._customers = {
            c.customer_id: c
            for c in (
                Customer("CUST-10001", "Ada Lovelace", "gold", Decimal("250.00")),
                Customer("CUST-10002", "Alan Turing", "silver", Decimal("75.50")),
            )
        }

    def get(self, customer_id: str) -> Customer:
        try:
            return self._customers[customer_id]
        except KeyError:
            raise CustomerNotFound(customer_id) from None

    def refund(self, customer_id: str, amount: float, reason: str) -> dict:
        customer = self.get(customer_id)
        value = Decimal(str(amount))
        if value > customer.refundable_balance:
            raise RefundExceedsBalance(customer_id)

        customer.refundable_balance -= value
        refund = {
            "refund_id": f"RFND-{uuid4().hex[:12]}",
            "customer_id": customer_id,
            "amount": amount,
            "reason": reason,
            "status": "processed",
        }
        customer.refunds.append(refund)
        return refund
