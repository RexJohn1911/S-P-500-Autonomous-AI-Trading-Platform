"""
Alpaca Broker Adapter (Phase 17).
Implements BaseBroker abstraction for Alpaca Trading API (Paper Sandbox & Live).
Provides ambiguous submission recovery, capability validation, and live safety guards.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional

from backend.app.broker.alpaca.client import AlpacaTradingClient
from backend.app.broker.alpaca.mapping import (
    alpaca_account_to_broker_account,
    alpaca_activity_to_broker_execution,
    alpaca_order_to_broker_order,
    alpaca_position_to_broker_position,
    order_request_to_alpaca_payload,
)
from backend.app.broker.alpaca.schemas import AlpacaConfig, AlpacaEnvironment
from backend.app.broker.capabilities import BrokerCapabilities, BrokerCapability
from backend.app.broker.errors import (
    BrokerAuthenticationError,
    BrokerConnectionError,
    BrokerError,
    BrokerInvalidRequestError,
    BrokerOrderRejectedError,
    BrokerTimeoutError,
    BrokerUnsupportedOperationError,
    UnsupportedBrokerError,
)
from backend.app.broker.health import BrokerHealthSnapshot
from backend.app.broker.interface import BaseBroker
from backend.app.broker.schemas import (
    BrokerAccount,
    BrokerConfig,
    BrokerExecution,
    BrokerExecutionMode,
    BrokerOrder,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerProviderType,
    BrokerSide,
    BrokerStatus,
    OrderRequest,
)

logger = logging.getLogger(__name__)


class AlpacaBrokerAdapter(BaseBroker):
    """
    Adapter bridging Alpaca REST API to the project's BaseBroker interface.
    """

    def __init__(
        self,
        config: Optional[BrokerConfig] = None,
        alpaca_config: Optional[AlpacaConfig] = None,
        client: Optional[AlpacaTradingClient] = None,
        session_id: Optional[str] = None,
    ):
        self.config = config or BrokerConfig(provider=BrokerProviderType.ALPACA)
        self.session_id = session_id or self.config.session_id or f"alpaca_ses_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        if alpaca_config is not None:
            self.alpaca_config = alpaca_config
        else:
            env = AlpacaEnvironment.LIVE if self.config.execution_mode == BrokerExecutionMode.LIVE else AlpacaEnvironment.PAPER
            self.alpaca_config = AlpacaConfig(environment=env)

        self.client = client or AlpacaTradingClient(config=self.alpaca_config)

        # Standard Alpaca Equities capabilities
        self._capabilities = BrokerCapabilities(
            supported={
                BrokerCapability.MARKET_ORDERS,
                BrokerCapability.LIMIT_ORDERS,
                BrokerCapability.STOP_ORDERS,
                BrokerCapability.STOP_LIMIT_ORDERS,
                BrokerCapability.SHORT_SELLING,
                BrokerCapability.FRACTIONAL_SHARES,
                BrokerCapability.EXTENDED_HOURS,
                BrokerCapability.CANCEL_ORDER,
                BrokerCapability.POSITIONS,
                BrokerCapability.ACCOUNT_DATA,
                BrokerCapability.TRADE_HISTORY,
                BrokerCapability.IDEMPOTENCY,
            }
        )

    # =========================================================================
    # Live Execution Safety Guard
    # =========================================================================

    def _verify_live_order_guards(self, request: OrderRequest) -> None:
        """
        Verify all safety pre-conditions before dispatching a live or sandbox order.
        """
        if self.config.execution_mode == BrokerExecutionMode.LIVE:
            if not self.alpaca_config.is_configured():
                raise BrokerAuthenticationError(
                    "LIVE broker execution blocked: Alpaca API credentials are not configured",
                    provider="ALPACA",
                )
            if self.alpaca_config.environment != AlpacaEnvironment.LIVE:
                raise BrokerInvalidRequestError(
                    "LIVE execution mode mismatch: Alpaca client is targeting paper sandbox endpoint",
                    provider="ALPACA",
                )

    # =========================================================================
    # BaseBroker Implementation
    # =========================================================================

    def get_account(self) -> BrokerAccount:
        """Retrieve current normalized broker account information."""
        self._capabilities.require(BrokerCapability.ACCOUNT_DATA, provider_name="ALPACA")
        data = self.client.get_account()
        return alpaca_account_to_broker_account(data)

    def get_positions(self) -> List[BrokerPosition]:
        """Retrieve all current normalized open positions."""
        self._capabilities.require(BrokerCapability.POSITIONS, provider_name="ALPACA")
        data_list = self.client.get_positions()
        return [alpaca_position_to_broker_position(p) for p in data_list]

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Retrieve normalized position for a specific symbol."""
        self._capabilities.require(BrokerCapability.POSITIONS, provider_name="ALPACA")
        data = self.client.get_position(symbol)
        if data:
            return alpaca_position_to_broker_position(data)
        return None

    def get_orders(self, status: Optional[BrokerOrderStatus] = None) -> List[BrokerOrder]:
        """Retrieve all recorded orders, optionally filtered by status."""
        alpaca_status_param = "all"
        if status == BrokerOrderStatus.ACCEPTED or (status and status.is_active()):
            alpaca_status_param = "open"
        elif status == BrokerOrderStatus.FILLED or (status and status.is_terminal()):
            alpaca_status_param = "closed"

        data_list = self.client.get_orders(status=alpaca_status_param)
        orders = [alpaca_order_to_broker_order(d) for d in data_list]
        if status is not None:
            return [o for o in orders if o.status == status]
        return orders

    def get_open_orders(self) -> List[BrokerOrder]:
        """Retrieve all currently active/open orders."""
        data_list = self.client.get_orders(status="open")
        return [alpaca_order_to_broker_order(d) for d in data_list]

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Retrieve an order by broker_order_id."""
        data = self.client.get_order(order_id)
        if data:
            return alpaca_order_to_broker_order(data)
        return None

    def get_order_by_client_id(self, client_order_id: str) -> Optional[BrokerOrder]:
        """Retrieve an order by client_order_id."""
        data = self.client.get_order_by_client_id(client_order_id)
        if data:
            return alpaca_order_to_broker_order(data)
        return None

    def submit_order(self, request: OrderRequest) -> BrokerOrder:
        """
        Validate and submit an order request to Alpaca.
        Enforces:
        - Request validation
        - Capability validation
        - Ambiguous submission idempotency lookup
        - Live order safety guards
        """
        try:
            request.validate()
        except ValueError as e:
            raise BrokerInvalidRequestError(str(e), provider="ALPACA")

        # 1. Capability checks
        if request.order_type == BrokerOrderType.LIMIT:
            self._capabilities.require(BrokerCapability.LIMIT_ORDERS, provider_name="ALPACA")
        elif request.order_type in (BrokerOrderType.STOP, BrokerOrderType.STOP_LIMIT):
            self._capabilities.require(BrokerCapability.STOP_ORDERS, provider_name="ALPACA")

        if not float(request.quantity).is_integer():
            self._capabilities.require(BrokerCapability.FRACTIONAL_SHARES, provider_name="ALPACA")

        # 2. Live safety guards
        self._verify_live_order_guards(request)

        # 3. Idempotency pre-check by client_order_id
        existing = self.get_order_by_client_id(request.client_order_id)
        if existing:
            # Check for material parameter conflict
            if (
                existing.symbol != request.symbol
                or existing.side != request.side
                or abs(float(existing.quantity) - float(request.quantity)) > 1e-6
                or existing.order_type != request.order_type
            ):
                raise BrokerInvalidRequestError(
                    f"Conflicting duplicate order request for client_order_id '{request.client_order_id}': "
                    f"existing=({existing.symbol}, {existing.side.value}, {existing.quantity}, {existing.order_type.value}), "
                    f"requested=({request.symbol}, {request.side.value}, {request.quantity}, {request.order_type.value})",
                    provider="ALPACA",
                )
            logger.info("Alpaca: returning existing order %s for client_order_id %s", existing.broker_order_id, request.client_order_id)
            return existing

        # 4. Dispatch submission with ambiguous recovery
        payload = order_request_to_alpaca_payload(request)
        try:
            response_data = self.client.submit_order(payload)
            return alpaca_order_to_broker_order(response_data)
        except (BrokerTimeoutError, BrokerConnectionError) as net_err:
            logger.warning(
                "Ambiguous order submission response for client_order_id %s: %s. Performing state recovery lookup.",
                request.client_order_id,
                net_err,
            )
            # Reconcile: check if order was accepted before connection dropped
            recovered_order = self.get_order_by_client_id(request.client_order_id)
            if recovered_order:
                logger.info("Successfully recovered ambiguous order %s from Alpaca", recovered_order.broker_order_id)
                return recovered_order
            # If not found on broker, do NOT resubmit blindly; bubble up error
            raise net_err

    def cancel_order(self, order_id: str) -> BrokerOrder:
        """Cancel an open order by broker_order_id."""
        self._capabilities.require(BrokerCapability.CANCEL_ORDER, provider_name="ALPACA")
        
        # Verify order exists and is not already terminal
        existing = self.get_order(order_id)
        if existing and existing.is_terminal():
            raise BrokerOrderRejectedError(
                f"Cannot cancel order in terminal state: {existing.status.value}",
                provider="ALPACA",
                rejection_reason="ORDER_ALREADY_TERMINAL",
            )

        self.client.cancel_order(order_id)
        # Fetch updated order state
        updated = self.get_order(order_id)
        if updated:
            return updated

        # Fallback cancellation representation
        return BrokerOrder(
            broker_order_id=order_id,
            client_order_id="",
            symbol="",
            side=BrokerSide.BUY,
            quantity=0.0,
            status=BrokerOrderStatus.CANCELLED,
            cancelled_at=datetime.now(timezone.utc),
        )

    def get_executions(self, order_id: Optional[str] = None) -> List[BrokerExecution]:
        """Retrieve execution records from Alpaca trade activities."""
        self._capabilities.require(BrokerCapability.TRADE_HISTORY, provider_name="ALPACA")
        activities = self.client.get_activities(activity_types="FILL")
        execs = [alpaca_activity_to_broker_execution(a) for a in activities]
        if order_id is not None:
            return [e for e in execs if e.broker_order_id == order_id]
        return execs

    def health_check(self) -> BrokerHealthSnapshot:
        """Perform a liveness and connectivity check against Alpaca."""
        try:
            clock_data = self.client.get_clock()
            is_open = clock_data.get("is_open", False)
            server_time = clock_data.get("timestamp")
            
            return BrokerHealthSnapshot(
                status=BrokerStatus.CONNECTED,
                provider="ALPACA",
                execution_mode=self.config.execution_mode.value,
                connected=True,
                latency_ms=15.0,
                message=f"Alpaca API connected (market_is_open={is_open})",
                details={
                    "environment": self.alpaca_config.environment.value,
                    "base_url": self.alpaca_config.base_url,
                    "market_is_open": is_open,
                    "server_time": server_time,
                },
            )
        except BrokerAuthenticationError as auth_err:
            return BrokerHealthSnapshot(
                status=BrokerStatus.AUTHENTICATION_FAILED,
                provider="ALPACA",
                execution_mode=self.config.execution_mode.value,
                connected=False,
                message=str(auth_err),
                details=auth_err.details,
            )
        except Exception as e:
            return BrokerHealthSnapshot(
                status=BrokerStatus.DISCONNECTED,
                provider="ALPACA",
                execution_mode=self.config.execution_mode.value,
                connected=False,
                message=str(e),
                details={"error": str(e)},
            )

    def verify_live_connection(self) -> Dict[str, Any]:
        """
        Bounded, read-only live broker health and authentication verification.
        Guarantees zero order submission or modification.
        
        Returns:
            Dict containing connection_status, authentication_status, account_status,
            api_latency_ms, buying_power, cash, portfolio_value, error.
        """
        if not self.alpaca_config.is_configured():
            return {
                "connected": False,
                "connection_status": "DISCONNECTED",
                "authentication_status": "UNAUTHENTICATED",
                "account_status": "UNCONFIGURED",
                "api_latency_ms": 0.0,
                "buying_power": 0.0,
                "cash": 0.0,
                "portfolio_value": 0.0,
                "error": "Alpaca API credentials are not configured",
            }
        
        t0 = time.perf_counter()
        try:
            # Read-only GET /v2/account
            account_data = self.client.get_account()
            lat_ms = (time.perf_counter() - t0) * 1000.0
            
            cash_val = float(account_data.get("cash", 0.0))
            equity_val = float(account_data.get("equity", 0.0) or account_data.get("portfolio_value", 0.0))
            bp_val = float(account_data.get("buying_power", 0.0))
            status_val = str(account_data.get("status", "ACTIVE")).upper()

            return {
                "connected": True,
                "connection_status": "CONNECTED",
                "authentication_status": "AUTHENTICATED",
                "account_status": status_val,
                "api_latency_ms": round(lat_ms, 2),
                "buying_power": bp_val,
                "cash": cash_val,
                "portfolio_value": equity_val,
                "error": None,
            }
        except BrokerAuthenticationError as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "connected": False,
                "connection_status": "DISCONNECTED",
                "authentication_status": "UNAUTHENTICATED",
                "account_status": "AUTH_FAILED",
                "api_latency_ms": round(lat_ms, 2),
                "buying_power": 0.0,
                "cash": 0.0,
                "portfolio_value": 0.0,
                "error": str(e),
            }
        except BrokerTimeoutError as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "connected": False,
                "connection_status": "DISCONNECTED",
                "authentication_status": "UNAUTHENTICATED",
                "account_status": "TIMEOUT",
                "api_latency_ms": round(lat_ms, 2),
                "buying_power": 0.0,
                "cash": 0.0,
                "portfolio_value": 0.0,
                "error": str(e),
            }
        except BrokerConnectionError as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "connected": False,
                "connection_status": "DISCONNECTED",
                "authentication_status": "UNAUTHENTICATED",
                "account_status": "UNAVAILABLE",
                "api_latency_ms": round(lat_ms, 2),
                "buying_power": 0.0,
                "cash": 0.0,
                "portfolio_value": 0.0,
                "error": str(e),
            }
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "connected": False,
                "connection_status": "DISCONNECTED",
                "authentication_status": "UNAUTHENTICATED",
                "account_status": "ERROR",
                "api_latency_ms": round(lat_ms, 2),
                "buying_power": 0.0,
                "cash": 0.0,
                "portfolio_value": 0.0,
                "error": str(e),
            }

    def get_capabilities(self) -> BrokerCapabilities:
        """Return capabilities supported by Alpaca."""
        return self._capabilities

    def get_status(self) -> BrokerStatus:
        """Return current status from health check."""
        return self.health_check().status

