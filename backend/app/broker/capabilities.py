"""
Broker Capability Model (Phase 16).
Defines standardized capability flags and verification methods.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Union

from backend.app.broker.errors import BrokerUnsupportedOperationError


class BrokerCapability(str, Enum):
    """Standardized broker capabilities."""
    MARKET_ORDERS = "MARKET_ORDERS"
    LIMIT_ORDERS = "LIMIT_ORDERS"
    STOP_ORDERS = "STOP_ORDERS"
    STOP_LIMIT_ORDERS = "STOP_LIMIT_ORDERS"
    SHORT_SELLING = "SHORT_SELLING"
    FRACTIONAL_SHARES = "FRACTIONAL_SHARES"
    EXTENDED_HOURS = "EXTENDED_HOURS"
    CANCEL_ORDER = "CANCEL_ORDER"
    ORDER_REPLACEMENT = "ORDER_REPLACEMENT"
    REAL_TIME_QUOTES = "REAL_TIME_QUOTES"
    POSITIONS = "POSITIONS"
    ACCOUNT_DATA = "ACCOUNT_DATA"
    TRADE_HISTORY = "TRADE_HISTORY"
    DIVIDENDS = "DIVIDENDS"
    CORPORATE_ACTIONS = "CORPORATE_ACTIONS"
    IDEMPOTENCY = "IDEMPOTENCY"


@dataclass
class BrokerCapabilities:
    """Container for broker capability inspection and enforcement."""
    supported: Set[BrokerCapability] = field(default_factory=set)

    def supports(self, capability: Union[BrokerCapability, str]) -> bool:
        """Check if capability is supported."""
        if isinstance(capability, str):
            try:
                cap = BrokerCapability(capability.upper())
            except ValueError:
                return False
        else:
            cap = capability
        return cap in self.supported

    def require(self, capability: Union[BrokerCapability, str], provider_name: str = "Broker") -> None:
        """Enforce capability; raise BrokerUnsupportedOperationError if not supported."""
        if not self.supports(capability):
            cap_name = capability.value if isinstance(capability, BrokerCapability) else str(capability)
            raise BrokerUnsupportedOperationError(
                f"Capability '{cap_name}' is not supported by {provider_name}",
                provider=provider_name,
            )

    def to_dict(self) -> Dict[str, bool]:
        """Return dict of all capabilities and boolean flags."""
        return {cap.value: (cap in self.supported) for cap in BrokerCapability}

    def to_list(self) -> List[str]:
        """Return sorted list of supported capability names."""
        return sorted([cap.value for cap in self.supported])
