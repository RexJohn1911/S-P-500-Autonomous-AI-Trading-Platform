"""
Base Broker Abstract Interface (Phase 16).
Defines the standardized provider-independent execution contract.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from backend.app.broker.capabilities import BrokerCapabilities
from backend.app.broker.health import BrokerHealthSnapshot
from backend.app.broker.schemas import (
    BrokerAccount,
    BrokerExecution,
    BrokerOrder,
    BrokerOrderStatus,
    BrokerPosition,
    BrokerStatus,
    OrderRequest,
)


class BaseBroker(ABC):
    """
    Abstract interface for all brokerage implementations.
    Separates the trading system's intent from provider-specific execution mechanics.
    """

    @abstractmethod
    def get_account(self) -> BrokerAccount:
        """Retrieve current normalized broker account information."""
        pass

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """Retrieve all current normalized open positions."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Retrieve normalized position for a specific symbol."""
        pass

    @abstractmethod
    def get_orders(self, status: Optional[BrokerOrderStatus] = None) -> List[BrokerOrder]:
        """Retrieve all recorded orders, optionally filtered by status."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[BrokerOrder]:
        """Retrieve all currently active/open orders."""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Retrieve an order by broker_order_id."""
        pass

    @abstractmethod
    def get_order_by_client_id(self, client_order_id: str) -> Optional[BrokerOrder]:
        """Retrieve an order by client_order_id."""
        pass

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> BrokerOrder:
        """Validate and submit an order request."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> BrokerOrder:
        """Cancel an open order by broker_order_id."""
        pass

    @abstractmethod
    def get_executions(self, order_id: Optional[str] = None) -> List[BrokerExecution]:
        """Retrieve execution records, optionally filtered by broker_order_id."""
        pass

    @abstractmethod
    def health_check(self) -> BrokerHealthSnapshot:
        """Perform a liveness and status check."""
        pass

    @abstractmethod
    def get_capabilities(self) -> BrokerCapabilities:
        """Return the capabilities supported by this broker."""
        pass

    @abstractmethod
    def get_status(self) -> BrokerStatus:
        """Return current operational status."""
        pass
