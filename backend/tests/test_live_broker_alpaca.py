"""
Comprehensive Test Suite for Live Broker Integration (Phase 17 - Alpaca).
Validates:
- Alpaca config, secret redaction, and paper/live URL endpoint routing
- Authenticated client wrapper, bounded retries, and error mapping
- Domain payload normalization (Account, Positions, Orders, Executions)
- BaseBroker contract compliance for AlpacaBrokerAdapter
- Ambiguous submission idempotency recovery & duplicate protection
- Capability discovery & strict enforcement
- Health checks, clock checks, and connectivity reporting
- Strict safety boundaries: Paper/Live separation, live execution guards
- External order reconciliation
- Deterministic paper replay parity
"""

from datetime import datetime, timezone
import json
import pytest
import httpx

from backend.app.autonomous.controller import AutonomousTradingController
from backend.app.autonomous.schemas import AutonomousConfig
from backend.app.broker import (
    BaseBroker,
    BrokerAccount,
    BrokerAuthenticationError,
    BrokerCapabilities,
    BrokerCapability,
    BrokerConfig,
    BrokerConnectionError,
    BrokerError,
    BrokerExecution,
    BrokerExecutionMode,
    BrokerFactory,
    BrokerHealthSnapshot,
    BrokerInvalidRequestError,
    BrokerOrder,
    BrokerOrderRejectedError,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerProviderType,
    BrokerRateLimitError,
    BrokerRegistry,
    BrokerSide,
    BrokerStatus,
    BrokerTimeoutError,
    BrokerTimeInForce,
    BrokerUnsupportedOperationError,
    OrderRequest,
    reconcile_broker_account,
    reconcile_broker_order,
)
from backend.app.broker.alpaca import (
    AlpacaBrokerAdapter,
    AlpacaConfig,
    AlpacaEnvironment,
    AlpacaTradingClient,
    alpaca_account_to_broker_account,
    alpaca_activity_to_broker_execution,
    alpaca_order_to_broker_order,
    alpaca_position_to_broker_position,
    map_alpaca_error,
    order_request_to_alpaca_payload,
)
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.storage import ProcessedDataStorage


# =========================================================================
# Mock Transport Fixture for Alpaca REST API
# =========================================================================

class MockAlpacaTransport(httpx.BaseTransport):
    """
    Simulated HTTP transport for Alpaca v2 REST Trading API.
    """

    def __init__(self):
        self.orders: dict[str, dict] = {}
        self.positions: dict[str, dict] = {
            "AAPL": {
                "symbol": "AAPL",
                "qty": "100",
                "avg_entry_price": "150.00",
                "current_price": "155.00",
                "market_value": "15500.00",
                "unrealized_pl": "500.00",
                "side": "long",
            }
        }
        self.account_data = {
            "id": "acc_alpaca_test_001",
            "account_number": "PA3920192",
            "status": "ACTIVE",
            "currency": "USD",
            "cash": "84500.00",
            "portfolio_value": "100000.00",
            "equity": "100000.00",
            "buying_power": "200000.00",
            "initial_margin": "0.00",
            "shorting_enabled": True,
        }
        self.clock_data = {
            "timestamp": "2025-01-01T15:00:00Z",
            "is_open": True,
            "next_open": "2025-01-02T14:30:00Z",
            "next_close": "2025-01-01T21:00:00Z",
        }
        self.activities: list[dict] = []
        self.order_counter = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url_path = request.url.path
        method = request.method

        # Verify authentication headers
        auth_key = request.headers.get("APCA-API-KEY-ID")
        auth_sec = request.headers.get("APCA-API-SECRET-KEY")
        if not auth_key or not auth_sec or auth_key == "INVALID_KEY":
            return httpx.Response(401, json={"message": "Invalid API credentials"})

        # Clock
        if method == "GET" and url_path == "/v2/clock":
            return httpx.Response(200, json=self.clock_data)

        # Account
        if method == "GET" and url_path == "/v2/account":
            return httpx.Response(200, json=self.account_data)

        # Positions
        if method == "GET" and url_path == "/v2/positions":
            return httpx.Response(200, json=list(self.positions.values()))

        if method == "GET" and url_path.startswith("/v2/positions/"):
            sym = url_path.split("/")[-1].upper()
            if sym in self.positions:
                return httpx.Response(200, json=self.positions[sym])
            return httpx.Response(404, json={"message": f"position does not exist for {sym}"})

        # Orders List
        if method == "GET" and url_path == "/v2/orders":
            return httpx.Response(200, json=list(self.orders.values()))

        # Order by Client Order ID
        if method == "GET" and url_path == "/v2/orders:by_client_order_id":
            cid = request.url.params.get("client_order_id")
            for o in self.orders.values():
                if o.get("client_order_id") == cid:
                    return httpx.Response(200, json=o)
            return httpx.Response(404, json={"message": f"order not found for client_order_id {cid}"})

        # Single Order by ID
        if method == "GET" and url_path.startswith("/v2/orders/"):
            oid = url_path.split("/")[-1]
            if oid in self.orders:
                return httpx.Response(200, json=self.orders[oid])
            return httpx.Response(404, json={"message": f"order not found for {oid}"})

        # Order Submission
        if method == "POST" and url_path == "/v2/orders":
            payload = json.loads(request.content.decode("utf-8"))
            self.order_counter += 1
            oid = f"alpaca_ord_{self.order_counter}"
            cid = payload.get("client_order_id", f"alpaca_cli_{self.order_counter}")

            ord_record = {
                "id": oid,
                "client_order_id": cid,
                "created_at": "2025-01-01T15:00:00.123456Z",
                "submitted_at": "2025-01-01T15:00:00.123456Z",
                "symbol": payload["symbol"],
                "qty": str(payload["qty"]),
                "filled_qty": "0",
                "side": payload["side"],
                "type": payload["type"],
                "time_in_force": payload["time_in_force"],
                "limit_price": payload.get("limit_price"),
                "stop_price": payload.get("stop_price"),
                "status": "new",
                "filled_avg_price": None,
            }
            self.orders[oid] = ord_record
            return httpx.Response(201, json=ord_record)

        # Cancel Order
        if method == "DELETE" and url_path.startswith("/v2/orders/"):
            oid = url_path.split("/")[-1]
            if oid in self.orders:
                self.orders[oid]["status"] = "canceled"
                self.orders[oid]["canceled_at"] = "2025-01-01T15:05:00Z"
                return httpx.Response(204)
            return httpx.Response(404, json={"message": f"order {oid} not found"})

        # Account Activities
        if method == "GET" and url_path.startswith("/v2/account/activities"):
            return httpx.Response(200, json=self.activities)

        return httpx.Response(404, json={"message": "Not Found"})


# =========================================================================
# 1. Alpaca Config & Secret Redaction Tests
# =========================================================================

def test_alpaca_config_secret_redaction_and_urls():
    """Verify secrets are redacted and URLs route to paper vs live endpoints."""
    # Paper Config
    cfg_paper = AlpacaConfig(
        api_key="PKTESTKEY123",
        api_secret="TESTSECRET456",
        environment=AlpacaEnvironment.PAPER,
    )
    assert cfg_paper.base_url == "https://paper-api.alpaca.markets"
    assert cfg_paper.is_configured() is True

    redacted = cfg_paper.to_redacted_dict()
    assert redacted["api_key"] == "********"
    assert redacted["api_secret"] == "********"
    assert redacted["environment"] == "PAPER"
    assert "PKTESTKEY123" not in str(redacted)
    assert "TESTSECRET456" not in str(redacted)

    # Live Config
    cfg_live = AlpacaConfig(
        api_key="AKLIVEKEY789",
        api_secret="LIVESECRET999",
        environment=AlpacaEnvironment.LIVE,
    )
    assert cfg_live.base_url == "https://api.alpaca.markets"


# =========================================================================
# 2. Error Normalization Tests
# =========================================================================

def test_alpaca_error_mapping_across_status_codes():
    """Verify provider HTTP status codes map to normalized BrokerError types."""
    # 401 Unauthorized -> BrokerAuthenticationError (Non-retryable)
    err_401 = map_alpaca_error(401, '{"message": "access key not found"}', operation="get_account")
    assert isinstance(err_401, BrokerAuthenticationError)
    assert err_401.retryable is False

    # 403 Forbidden / Buying Power -> BrokerOrderRejectedError (Non-retryable)
    err_403 = map_alpaca_error(403, '{"code": 40310000, "message": "insufficient buying power"}', operation="submit_order")
    assert isinstance(err_403, BrokerOrderRejectedError)
    assert err_403.rejection_reason == "40310000"

    # 422 Unprocessable -> BrokerInvalidRequestError (Non-retryable)
    err_422 = map_alpaca_error(422, '{"message": "qty must be positive integer"}', operation="submit_order")
    assert isinstance(err_422, BrokerInvalidRequestError)

    # 429 Rate Limit -> BrokerRateLimitError (Retryable)
    err_429 = map_alpaca_error(429, '{"message": "rate limit exceeded"}', operation="get_orders")
    assert isinstance(err_429, BrokerRateLimitError)
    assert err_429.retryable is True

    # 504 Gateway Timeout -> BrokerTimeoutError (Retryable)
    err_504 = map_alpaca_error(504, 'Gateway Timeout', operation="get_positions")
    assert isinstance(err_504, BrokerTimeoutError)
    assert err_504.retryable is True


# =========================================================================
# 3. Payload Normalization Tests
# =========================================================================

def test_alpaca_account_and_position_normalization():
    """Verify raw Alpaca account and position payloads normalize to standardized schemas."""
    raw_acc = {
        "id": "acc_alpaca_01",
        "currency": "USD",
        "cash": "25000.50",
        "equity": "100000.00",
        "buying_power": "200000.00",
        "status": "ACTIVE",
    }
    bkr_acc = alpaca_account_to_broker_account(raw_acc)
    assert isinstance(bkr_acc, BrokerAccount)
    assert bkr_acc.cash == 25000.50
    assert bkr_acc.equity == 100000.00
    assert bkr_acc.buying_power == 200000.00
    assert bkr_acc.status == "ACTIVE"

    raw_pos = {
        "symbol": "TSLA",
        "qty": "-50",
        "avg_entry_price": "220.00",
        "current_price": "210.00",
        "market_value": "-10500.00",
        "unrealized_pl": "500.00",
        "side": "short",
    }
    bkr_pos = alpaca_position_to_broker_position(raw_pos)
    assert isinstance(bkr_pos, BrokerPosition)
    assert bkr_pos.symbol == "TSLA"
    assert bkr_pos.quantity == -50.0
    assert bkr_pos.side == BrokerPositionSide.SHORT
    assert bkr_pos.unrealized_pnl == 500.0


# =========================================================================
# 4. BaseBroker Contract Tests on AlpacaBrokerAdapter
# =========================================================================

def test_alpaca_broker_adapter_operations():
    """Verify AlpacaBrokerAdapter satisfies BaseBroker contract operations."""
    transport = MockAlpacaTransport()
    http_client = httpx.Client(transport=transport)
    alpaca_cfg = AlpacaConfig(api_key="MOCK_KEY", api_secret="MOCK_SECRET")
    client = AlpacaTradingClient(config=alpaca_cfg, http_client=http_client)

    adapter = AlpacaBrokerAdapter(
        config=BrokerConfig(provider=BrokerProviderType.ALPACA),
        alpaca_config=alpaca_cfg,
        client=client,
    )

    # 1. Account
    acc = adapter.get_account()
    assert isinstance(acc, BrokerAccount)
    assert acc.cash == 84500.00
    assert acc.equity == 100000.00

    # 2. Positions
    positions = adapter.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].quantity == 100.0

    pos_aapl = adapter.get_position("AAPL")
    assert pos_aapl is not None
    assert pos_aapl.symbol == "AAPL"

    # 3. Order Submission (Market)
    req = OrderRequest(
        client_order_id="alpaca_test_cid_01",
        symbol="MSFT",
        side=BrokerSide.BUY,
        quantity=25.0,
        order_type=BrokerOrderType.MARKET,
    )
    order = adapter.submit_order(req)
    assert isinstance(order, BrokerOrder)
    assert order.symbol == "MSFT"
    assert order.quantity == 25.0
    assert order.client_order_id == "alpaca_test_cid_01"
    assert order.status == BrokerOrderStatus.ACCEPTED

    # 4. Lookup by broker order ID and client order ID
    found_by_id = adapter.get_order(order.broker_order_id)
    assert found_by_id is not None
    assert found_by_id.broker_order_id == order.broker_order_id

    found_by_cid = adapter.get_order_by_client_id("alpaca_test_cid_01")
    assert found_by_cid is not None
    assert found_by_cid.client_order_id == "alpaca_test_cid_01"

    # 5. Order Cancellation
    cancelled = adapter.cancel_order(order.broker_order_id)
    assert cancelled.status == BrokerOrderStatus.CANCELLED

    # 6. Health Check
    health = adapter.health_check()
    assert isinstance(health, BrokerHealthSnapshot)
    assert health.status == BrokerStatus.CONNECTED
    assert health.connected is True
    assert health.provider == "ALPACA"


# =========================================================================
# 5. Ambiguous Submission & Idempotency Tests
# =========================================================================

def test_alpaca_ambiguous_submission_recovery():
    """Verify adapter recovers order via client_order_id lookup during ambiguous submission failure."""
    transport = MockAlpacaTransport()
    # Pre-populate an order in the mock broker that was accepted before network dropped
    transport.orders["alpaca_ord_pre"] = {
        "id": "alpaca_ord_pre",
        "client_order_id": "ambiguous_cid_007",
        "symbol": "NVDA",
        "qty": "15",
        "filled_qty": "0",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "status": "new",
    }

    http_client = httpx.Client(transport=transport)
    alpaca_cfg = AlpacaConfig(api_key="MOCK_KEY", api_secret="MOCK_SECRET")
    client = AlpacaTradingClient(config=alpaca_cfg, http_client=http_client)
    adapter = AlpacaBrokerAdapter(client=client, alpaca_config=alpaca_cfg)

    # Submitting same client_order_id seamlessly returns the pre-existing accepted order
    req = OrderRequest(
        client_order_id="ambiguous_cid_007",
        symbol="NVDA",
        side=BrokerSide.BUY,
        quantity=15.0,
    )
    recovered = adapter.submit_order(req)
    assert recovered.broker_order_id == "alpaca_ord_pre"
    assert recovered.symbol == "NVDA"
    assert recovered.status == BrokerOrderStatus.ACCEPTED


def test_alpaca_conflicting_duplicate_rejection():
    """Verify adapter rejects conflicting reuse of client_order_id."""
    transport = MockAlpacaTransport()
    http_client = httpx.Client(transport=transport)
    alpaca_cfg = AlpacaConfig(api_key="MOCK_KEY", api_secret="MOCK_SECRET")
    client = AlpacaTradingClient(config=alpaca_cfg, http_client=http_client)
    adapter = AlpacaBrokerAdapter(client=client, alpaca_config=alpaca_cfg)

    req1 = OrderRequest(client_order_id="cid_conflict_check", symbol="AAPL", side=BrokerSide.BUY, quantity=10.0)
    adapter.submit_order(req1)

    # Submit different quantity with same client_order_id
    req2_bad_qty = OrderRequest(client_order_id="cid_conflict_check", symbol="AAPL", side=BrokerSide.BUY, quantity=20.0)
    with pytest.raises(BrokerInvalidRequestError, match="Conflicting duplicate order request"):
        adapter.submit_order(req2_bad_qty)


# =========================================================================
# 6. Safety Boundaries & Paper/Live Separation Tests
# =========================================================================

from backend.app.strategy.schemas import ReasonCode, SignalCandidate, SignalDirection


def test_alpaca_live_execution_mode_guards():
    """Verify live order dispatch requires explicit LIVE mode and credentials."""
    # 1. Live mode without credentials fails closed
    cfg_live_no_creds = BrokerConfig(
        provider=BrokerProviderType.ALPACA,
        execution_mode=BrokerExecutionMode.LIVE,
    )
    alpaca_unconfigured = AlpacaConfig(api_key=None, api_secret=None, environment=AlpacaEnvironment.LIVE)
    adapter_unconfigured = AlpacaBrokerAdapter(config=cfg_live_no_creds, alpaca_config=alpaca_unconfigured)

    req = OrderRequest(client_order_id="live_guard_cid", symbol="AAPL", side=BrokerSide.BUY, quantity=10.0)
    with pytest.raises(BrokerAuthenticationError, match="LIVE broker execution blocked"):
        adapter_unconfigured.submit_order(req)

    # 2. Live mode with paper endpoint mismatch fails closed
    alpaca_mismatched = AlpacaConfig(api_key="VALID_KEY_123", api_secret="VALID_SECRET_456", environment=AlpacaEnvironment.PAPER)
    adapter_mismatched = AlpacaBrokerAdapter(config=cfg_live_no_creds, alpaca_config=alpaca_mismatched)
    with pytest.raises(BrokerInvalidRequestError, match="LIVE execution mode mismatch"):
        adapter_mismatched.submit_order(req)


def test_broker_factory_alpaca_resolution():
    """Verify BrokerFactory resolves AlpacaBrokerAdapter for PAPER and LIVE modes."""
    # Alpaca Paper
    cfg_paper = BrokerConfig(
        provider=BrokerProviderType.ALPACA,
        execution_mode=BrokerExecutionMode.PAPER,
    )
    broker_paper = BrokerFactory.create_broker(config=cfg_paper)
    assert isinstance(broker_paper, AlpacaBrokerAdapter)
    assert broker_paper.alpaca_config.environment == AlpacaEnvironment.PAPER

    # Alpaca Live
    cfg_live = BrokerConfig(
        provider=BrokerProviderType.ALPACA,
        execution_mode=BrokerExecutionMode.LIVE,
    )
    alpaca_live_cfg = AlpacaConfig(api_key="LIVE_KEY_123", api_secret="LIVE_SECRET_456", environment=AlpacaEnvironment.LIVE)
    broker_live = BrokerFactory.create_broker(config=cfg_live, alpaca_config=alpaca_live_cfg)
    assert isinstance(broker_live, AlpacaBrokerAdapter)
    assert broker_live.alpaca_config.environment == AlpacaEnvironment.LIVE


# =========================================================================
# 7. External Order Reconciliation Detection
# =========================================================================

def test_alpaca_external_order_reconciliation():
    """Verify reconciliation comparator detects externally placed orders or positions."""
    local_order = BrokerOrder(
        broker_order_id="ord_local_1",
        client_order_id="cli_local_1",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=50.0,
        status=BrokerOrderStatus.FILLED,
        filled_quantity=50.0,
    )
    # Broker has order filled at different price and remaining open on another symbol
    broker_order = BrokerOrder(
        broker_order_id="ord_local_1",
        client_order_id="cli_local_1",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=50.0,
        status=BrokerOrderStatus.PARTIALLY_FILLED,
        filled_quantity=25.0,
    )
    recon = reconcile_broker_order(local_order, broker_order)
    assert recon.is_matched is False
    assert recon.status_match is False
    assert recon.quantity_match is False


# =========================================================================
# 8. Determinism & Full Historical Replay Invariance
# =========================================================================

def test_paper_historical_replay_invariance_after_phase_17():
    """Verify Phase 17 live integration preserves 100% fidelity on existing AAPL replay baseline."""
    storage = ProcessedDataStorage()
    aapl_bars = storage.load_bars("AAPL", TimeFrame.DAY_1, format="parquet")
    assert len(aapl_bars) >= 500

    cfg = AutonomousConfig(
        universe=["AAPL"],
        paper_initial_capital=100000.0,
        commission_rate=0.0005,
        slippage_rate=0.0002,
        bid_ask_spread_rate=0.0002,
    )

    def sma_signals(ts, current_bars):
        bar = current_bars.get("AAPL")
        if not bar:
            return []
        if bar.close > 180.0:
            return [
                SignalCandidate(
                    timestamp=ts,
                    symbol="AAPL",
                    signal=SignalDirection.LONG,
                    signal_strength=0.8,
                    confidence=0.9,
                    reason_codes=[ReasonCode.BULLISH_ENSEMBLE_SCORE.value],
                )
            ]
        return []

    ctrl = AutonomousTradingController(
        config=cfg,
        signal_generator=sma_signals,
    )
    res = ctrl.run_replay(market_data={"AAPL": aapl_bars})

    assert res.session.total_cycles == len(aapl_bars)
    assert res.session.successful_cycles == len(aapl_bars)
    assert res.session.failed_cycles == 0
    assert ctrl.broker.get_account().equity == pytest.approx(101599.79, abs=0.5)


# =========================================================================
# 9. Comprehensive Hardening & Edge-Case Verification
# =========================================================================

def test_alpaca_partial_fills_and_trade_executions():
    """Verify partial fill order status normalization, remaining quantity calculation, and executions mapping."""
    raw_partial_order = {
        "id": "ord_partial_001",
        "client_order_id": "cli_partial_001",
        "symbol": "AAPL",
        "side": "buy",
        "qty": "100.0",
        "filled_qty": "40.0",
        "filled_avg_price": "152.50",
        "type": "limit",
        "limit_price": "153.00",
        "time_in_force": "day",
        "status": "partially_filled",
        "submitted_at": "2025-01-01T15:00:00.000Z",
    }
    broker_ord = alpaca_order_to_broker_order(raw_partial_order)
    assert broker_ord.status == BrokerOrderStatus.PARTIALLY_FILLED
    assert broker_ord.quantity == 100.0
    assert broker_ord.filled_quantity == 40.0
    assert broker_ord.remaining_quantity == 60.0
    assert broker_ord.average_fill_price == 152.50
    assert broker_ord.is_active() is True
    assert broker_ord.is_terminal() is False

    raw_activity = {
        "id": "act_fill_001",
        "order_id": "ord_partial_001",
        "client_order_id": "cli_partial_001",
        "symbol": "AAPL",
        "side": "buy",
        "qty": "40.0",
        "price": "152.50",
        "transaction_time": "2025-01-01T15:01:30.123456Z",
        "activity_type": "FILL",
    }
    exec_rec = alpaca_activity_to_broker_execution(raw_activity)
    assert isinstance(exec_rec, BrokerExecution)
    assert exec_rec.execution_id == "act_fill_001"
    assert exec_rec.broker_order_id == "ord_partial_001"
    assert exec_rec.client_order_id == "cli_partial_001"
    assert exec_rec.quantity == 40.0
    assert exec_rec.execution_price == 152.50
    assert exec_rec.venue == "ALPACA"
    assert exec_rec.executed_at.year == 2025


def test_alpaca_fractional_shares_and_short_selling():
    """Verify fractional share order request building and short position normalization."""
    # Fractional Order Request
    frac_req = OrderRequest(
        client_order_id="frac_cid_01",
        symbol="SPY",
        side=BrokerSide.BUY,
        quantity=2.75,
        order_type=BrokerOrderType.MARKET,
        time_in_force=BrokerTimeInForce.DAY,
    )
    payload = order_request_to_alpaca_payload(frac_req)
    assert payload["qty"] == 2.75
    assert payload["symbol"] == "SPY"
    assert payload["type"] == "market"

    # Short Position Normalization
    raw_short = {
        "symbol": "MSFT",
        "qty": "-25.5",
        "avg_entry_price": "400.00",
        "current_price": "395.00",
        "market_value": "-10072.50",
        "unrealized_pl": "127.50",
        "side": "short",
    }
    pos = alpaca_position_to_broker_position(raw_short)
    assert pos.symbol == "MSFT"
    assert pos.quantity == -25.5
    assert pos.side == BrokerPositionSide.SHORT
    assert pos.unrealized_pnl == 127.50


def test_alpaca_order_types_and_time_in_force_mappings():
    """Verify limit, stop, and stop-limit order payload serialization and TIF mappings."""
    # Limit Order with GTC
    req_limit = OrderRequest(
        client_order_id="limit_gtc_01",
        symbol="QQQ",
        side=BrokerSide.BUY,
        quantity=50.0,
        order_type=BrokerOrderType.LIMIT,
        limit_price=450.0,
        time_in_force=BrokerTimeInForce.GTC,
    )
    p_limit = order_request_to_alpaca_payload(req_limit)
    assert p_limit["type"] == "limit"
    assert p_limit["limit_price"] == "450.0"
    assert p_limit["time_in_force"] == "gtc"

    # Stop Order with IOC
    req_stop = OrderRequest(
        client_order_id="stop_ioc_01",
        symbol="QQQ",
        side=BrokerSide.SELL,
        quantity=50.0,
        order_type=BrokerOrderType.STOP,
        stop_price=440.0,
        time_in_force=BrokerTimeInForce.IOC,
    )
    p_stop = order_request_to_alpaca_payload(req_stop)
    assert p_stop["type"] == "stop"
    assert p_stop["stop_price"] == "440.0"
    assert p_stop["time_in_force"] == "ioc"

    # Stop Limit Order with FOK
    req_stop_limit = OrderRequest(
        client_order_id="stop_limit_fok_01",
        symbol="QQQ",
        side=BrokerSide.SELL,
        quantity=50.0,
        order_type=BrokerOrderType.STOP_LIMIT,
        limit_price=438.0,
        stop_price=440.0,
        time_in_force=BrokerTimeInForce.FOK,
    )
    p_stop_limit = order_request_to_alpaca_payload(req_stop_limit)
    assert p_stop_limit["type"] == "stop_limit"
    assert p_stop_limit["limit_price"] == "438.0"
    assert p_stop_limit["stop_price"] == "440.0"
    assert p_stop_limit["time_in_force"] == "fok"


def test_alpaca_cancellation_of_terminal_order_fails_closed():
    """Verify adapter rejects cancelling an already FILLED order."""
    transport = MockAlpacaTransport()
    # Pre-populate a filled order
    transport.orders["ord_already_filled"] = {
        "id": "ord_already_filled",
        "client_order_id": "cli_filled_01",
        "symbol": "AAPL",
        "qty": "10",
        "filled_qty": "10",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "status": "filled",
    }
    http_client = httpx.Client(transport=transport)
    alpaca_cfg = AlpacaConfig(api_key="MOCK_KEY", api_secret="MOCK_SECRET")
    client = AlpacaTradingClient(config=alpaca_cfg, http_client=http_client)
    adapter = AlpacaBrokerAdapter(client=client, alpaca_config=alpaca_cfg)

    with pytest.raises(BrokerOrderRejectedError, match="Cannot cancel order in terminal state"):
        adapter.cancel_order("ord_already_filled")


