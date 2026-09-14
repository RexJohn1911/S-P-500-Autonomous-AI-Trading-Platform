"""
Broker Registry (Phase 16).
Manages provider and execution-mode registration.
Fails closed with zero silent fallback.
"""

from typing import Callable, Dict, Optional, Tuple, Type

from backend.app.broker.alpaca.adapter import AlpacaBrokerAdapter
from backend.app.broker.errors import UnsupportedBrokerError
from backend.app.broker.interface import BaseBroker
from backend.app.broker.paper import PaperBrokerAdapter
from backend.app.broker.schemas import BrokerExecutionMode, BrokerProviderType


class BrokerRegistry:
    """
    Central registry for broker implementations.
    """

    _registry: Dict[Tuple[BrokerProviderType, BrokerExecutionMode], Type[BaseBroker]] = {}

    @classmethod
    def register(
        cls,
        provider: BrokerProviderType,
        mode: BrokerExecutionMode,
        broker_cls: Type[BaseBroker],
    ) -> None:
        """Register a broker implementation for a (provider, mode) pair."""
        cls._registry[(provider, mode)] = broker_cls

    @classmethod
    def get(
        cls,
        provider: BrokerProviderType,
        mode: BrokerExecutionMode,
    ) -> Type[BaseBroker]:
        """
        Lookup broker class. Raises UnsupportedBrokerError if not registered.
        """
        key = (provider, mode)
        if key not in cls._registry:
            raise UnsupportedBrokerError(
                f"No broker registered for provider='{provider.value}', mode='{mode.value}'. Fail-closed.",
                provider=provider.value,
            )
        return cls._registry[key]

    @classmethod
    def clear(cls) -> None:
        """Reset registry (primarily for testing)."""
        cls._registry.clear()
        cls._register_defaults()

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default Phase 16 and Phase 17 implementations."""
        cls._registry[(BrokerProviderType.PAPER, BrokerExecutionMode.PAPER)] = PaperBrokerAdapter
        cls._registry[(BrokerProviderType.ALPACA, BrokerExecutionMode.PAPER)] = AlpacaBrokerAdapter
        cls._registry[(BrokerProviderType.ALPACA, BrokerExecutionMode.LIVE)] = AlpacaBrokerAdapter


# Initialize default registry
BrokerRegistry._register_defaults()
