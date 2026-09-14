"""
Broker Abstraction Layer (Phase 16).
Provides normalized schemas, standardized BaseBroker interfaces, capability models,
error hierarchy, PaperBrokerAdapter, and BrokerFactory.
"""

from backend.app.broker.capabilities import BrokerCapabilities, BrokerCapability
from backend.app.broker.errors import (
    BrokerAccountError,
    BrokerAuthenticationError,
    BrokerConnectionError,
    BrokerError,
    BrokerInvalidRequestError,
    BrokerOrderRejectedError,
    BrokerRateLimitError,
    BrokerTimeoutError,
    BrokerUnknownError,
    BrokerUnsupportedOperationError,
    UnsupportedBrokerError,
    is_broker_error_retryable,
)
from backend.app.broker.factory import BrokerFactory
from backend.app.broker.health import BrokerHealthSnapshot
from backend.app.broker.interface import BaseBroker
from backend.app.broker.paper import PaperBrokerAdapter
from backend.app.broker.registry import BrokerRegistry
from backend.app.broker.schemas import (
    AccountSnapshot,
    BrokerAccount,
    BrokerAccountReconciliation,
    BrokerConfig,
    BrokerExecution,
    BrokerExecutionMode,
    BrokerOrder,
    BrokerOrderReconciliation,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerProviderType,
    BrokerQuote,
    BrokerSide,
    BrokerStatus,
    BrokerTimeInForce,
    LEGAL_ORDER_STATUS_TRANSITIONS,
    OrderRequest,
    OrderUpdate,
    PositionSnapshot,
    reconcile_broker_account,
    reconcile_broker_order,
    validate_order_status_transition,
)

__all__ = [
    # Schemas & Enums
    "BrokerExecutionMode",
    "BrokerProviderType",
    "BrokerSide",
    "BrokerOrderType",
    "BrokerTimeInForce",
    "BrokerOrderStatus",
    "BrokerPositionSide",
    "BrokerStatus",
    "LEGAL_ORDER_STATUS_TRANSITIONS",
    "validate_order_status_transition",
    "OrderRequest",
    "OrderUpdate",
    "BrokerOrder",
    "BrokerExecution",
    "BrokerPosition",
    "BrokerAccount",
    "BrokerQuote",
    "PositionSnapshot",
    "AccountSnapshot",
    "BrokerConfig",
    "BrokerOrderReconciliation",
    "BrokerAccountReconciliation",
    "reconcile_broker_order",
    "reconcile_broker_account",
    # Capabilities
    "BrokerCapability",
    "BrokerCapabilities",
    # Errors
    "BrokerError",
    "BrokerConnectionError",
    "BrokerTimeoutError",
    "BrokerRateLimitError",
    "BrokerAuthenticationError",
    "BrokerOrderRejectedError",
    "BrokerUnsupportedOperationError",
    "UnsupportedBrokerError",
    "BrokerInvalidRequestError",
    "BrokerAccountError",
    "BrokerUnknownError",
    "is_broker_error_retryable",
    # Interfaces & Adapters
    "BaseBroker",
    "PaperBrokerAdapter",
    "BrokerHealthSnapshot",
    "BrokerRegistry",
    "BrokerFactory",
]
