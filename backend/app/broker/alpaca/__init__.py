"""
Alpaca Broker Package (Phase 17).
Provides live and paper sandbox REST integration for US equities via Alpaca v2.
"""

from backend.app.broker.alpaca.adapter import AlpacaBrokerAdapter
from backend.app.broker.alpaca.client import AlpacaTradingClient
from backend.app.broker.alpaca.errors import map_alpaca_error
from backend.app.broker.alpaca.mapping import (
    alpaca_account_to_broker_account,
    alpaca_activity_to_broker_execution,
    alpaca_order_to_broker_order,
    alpaca_position_to_broker_position,
    order_request_to_alpaca_payload,
)
from backend.app.broker.alpaca.schemas import AlpacaConfig, AlpacaEnvironment

__all__ = [
    "AlpacaEnvironment",
    "AlpacaConfig",
    "AlpacaTradingClient",
    "AlpacaBrokerAdapter",
    "map_alpaca_error",
    "alpaca_account_to_broker_account",
    "alpaca_position_to_broker_position",
    "alpaca_order_to_broker_order",
    "alpaca_activity_to_broker_execution",
    "order_request_to_alpaca_payload",
]
