"""
Comprehensive Test Suite for Broker Abstraction Layer (Phase 16).
Validates:
- Normalized contracts & strict request validation
- BaseBroker interface compliance & PaperBrokerAdapter
- Order lifecycle, query consistency & idempotency
- Capability inspection & enforcement
- Normalized error hierarchy & retry classification
- Factory / Registry resolution with fail-closed safety
- Live broker safety boundary (LIVE execution mode forbidden)
- Determinism & reconciliation
- Historical replay parity with Paper / Autonomous layers
"""

from datetime import datetime, timezone
import math
import pytest

from backend.app.autonomous.controller import AutonomousTradingController
from backend.app.autonomous.schemas import AutonomousConfig
from backend.app.broker import (
    BaseBroker,
    BrokerAccount,
    BrokerAccountReconciliation,
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
    BrokerOrderReconciliation,
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
    PaperBrokerAdapter,
    UnsupportedBrokerError,
    is_broker_error_retryable,
    reconcile_broker_account,
    reconcile_broker_order,
    validate_order_status_transition,
)
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.storage import ProcessedDataStorage
from backend.app.paper_trading.broker import SimulatedPaperBroker
from backend.app.paper_trading.schemas import PaperTradingConfig


# =========================================================================
# 1. Normalized Contracts & Request Validation
# =========================================================================

def test_order_request_validation_valid():
    """Verify valid OrderRequest instantiates and serializes properly."""
    t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    req = OrderRequest(
        client_order_id="client_ord_001",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=10.0,
        order_type=BrokerOrderType.LIMIT,
        limit_price=150.0,
        submitted_at=t0,
    )
    assert req.symbol == "AAPL"
    assert req.quantity == 10.0
    assert req.limit_price == 150.0
    assert req.order_type == BrokerOrderType.LIMIT

    d = req.to_dict()
    assert d["client_order_id"] == "client_ord_001"
    assert d["side"] == "BUY"
    assert d["order_type"] == "LIMIT"


def test_order_request_validation_invalid_fields():
    """Verify invalid order request parameters raise ValueError before reaching broker."""
    # 1. Empty client_order_id
    with pytest.raises(ValueError, match="client_order_id"):
        OrderRequest(client_order_id="", symbol="AAPL", side=BrokerSide.BUY, quantity=10.0)

    # 2. Empty symbol
    with pytest.raises(ValueError, match="symbol"):
        OrderRequest(client_order_id="cid_1", symbol="", side=BrokerSide.BUY, quantity=10.0)

    # 3. Non-positive or non-finite quantity
    with pytest.raises(ValueError, match="quantity"):
        OrderRequest(client_order_id="cid_1", symbol="AAPL", side=BrokerSide.BUY, quantity=0.0)
    with pytest.raises(ValueError, match="quantity"):
        OrderRequest(client_order_id="cid_1", symbol="AAPL", side=BrokerSide.BUY, quantity=-5.0)
    with pytest.raises(ValueError, match="quantity"):
        OrderRequest(client_order_id="cid_1", symbol="AAPL", side=BrokerSide.BUY, quantity=float("nan"))
    with pytest.raises(ValueError, match="quantity"):
        OrderRequest(client_order_id="cid_1", symbol="AAPL", side=BrokerSide.BUY, quantity=float("inf"))

    # 4. Missing limit_price for LIMIT order
    with pytest.raises(ValueError, match="limit_price is required"):
        OrderRequest(client_order_id="cid_1", symbol="AAPL", side=BrokerSide.BUY, quantity=10.0, order_type=BrokerOrderType.LIMIT)

    # 5. Missing stop_price for STOP order
    with pytest.raises(ValueError, match="stop_price is required"):
        OrderRequest(client_order_id="cid_1", symbol="AAPL", side=BrokerSide.BUY, quantity=10.0, order_type=BrokerOrderType.STOP)


# =========================================================================
# 2. BaseBroker Interface & PaperBrokerAdapter Compliance
# =========================================================================

def test_paper_broker_adapter_account_and_positions():
    """Verify PaperBrokerAdapter implements BaseBroker account and position operations."""
    cfg = BrokerConfig(initial_capital=50000.0, currency="USD")
    adapter = PaperBrokerAdapter(config=cfg)

    # 1. Account retrieval
    acc = adapter.get_account()
    assert isinstance(acc, BrokerAccount)
    assert acc.cash == 50000.0
    assert acc.equity == 50000.0
    assert acc.buying_power >= 50000.0
    assert acc.currency == "USD"
    assert acc.status == "ACTIVE"

    # 2. Positions retrieval (initially empty)
    positions = adapter.get_positions()
    assert len(positions) == 0
    assert adapter.get_position("AAPL") is None


def test_paper_broker_adapter_order_lifecycle():
    """Verify order submission, execution, lookup, and cancellation through PaperBrokerAdapter."""
    t0 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    adapter = PaperBrokerAdapter(config=BrokerConfig(initial_capital=100000.0))

    # 1. Submit Market Buy Order
    req = OrderRequest(
        client_order_id="req_test_001",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=50.0,
        order_type=BrokerOrderType.MARKET,
        submitted_at=t0,
    )
    order = adapter.submit_order(req)
    assert isinstance(order, BrokerOrder)
    assert order.client_order_id == "req_test_001"
    assert order.status == BrokerOrderStatus.ACCEPTED
    assert order.symbol == "AAPL"
    assert order.quantity == 50.0

    # 2. Query order by broker_order_id & client_order_id
    retrieved_by_id = adapter.get_order(order.broker_order_id)
    assert retrieved_by_id is not None
    assert retrieved_by_id.broker_order_id == order.broker_order_id

    retrieved_by_cid = adapter.get_order_by_client_id("req_test_001")
    assert retrieved_by_cid is not None
    assert retrieved_by_cid.client_order_id == "req_test_001"

    # 3. Execute Market Order fill
    exec_rec = adapter.execute_market_order(
        order_id=order.broker_order_id,
        market_price=150.0,
        timestamp=t0,
    )
    assert isinstance(exec_rec, BrokerExecution)
    assert exec_rec.execution_price > 0.0
    assert exec_rec.quantity == 50.0
    assert exec_rec.symbol == "AAPL"

    # 4. Verify post-execution position & account
    pos = adapter.get_position("AAPL")
    assert pos is not None
    assert pos.quantity == 50.0
    assert pos.side == BrokerPositionSide.LONG
    assert pos.average_price > 0.0

    acc = adapter.get_account()
    assert acc.cash < 100000.0  # Cash reduced by purchase cost + commission

    # 5. Executions list
    execs = adapter.get_executions()
    assert len(execs) == 1
    assert execs[0].execution_id == exec_rec.execution_id


def test_order_idempotency_in_paper_adapter():
    """Verify repeated submission of same client_order_id returns existing order without duplicates."""
    adapter = PaperBrokerAdapter(config=BrokerConfig(initial_capital=50000.0))
    req = OrderRequest(
        client_order_id="idempotent_key_123",
        symbol="MSFT",
        side=BrokerSide.BUY,
        quantity=20.0,
    )

    ord1 = adapter.submit_order(req)
    ord2 = adapter.submit_order(req)

    assert ord1.broker_order_id == ord2.broker_order_id
    assert len(adapter.get_orders()) == 1


def test_order_cancellation_lifecycle():
    """Verify cancellation of open order and rejection on terminal state."""
    adapter = PaperBrokerAdapter(config=BrokerConfig(initial_capital=50000.0))
    req = OrderRequest(
        client_order_id="cancel_test_001",
        symbol="GOOGL",
        side=BrokerSide.BUY,
        quantity=10.0,
    )
    ord1 = adapter.submit_order(req)
    assert ord1.status == BrokerOrderStatus.ACCEPTED

    # Cancel open order
    cancelled = adapter.cancel_order(ord1.broker_order_id)
    assert cancelled.status == BrokerOrderStatus.CANCELLED

    # Cancel already cancelled order raises BrokerOrderRejectedError
    with pytest.raises(BrokerOrderRejectedError):
        adapter.cancel_order(ord1.broker_order_id)


# =========================================================================
# 3. Capability Model
# =========================================================================

def test_broker_capabilities_inspection_and_enforcement():
    """Verify capability inspection, requirement assertion, and unsupported error raising."""
    caps = BrokerCapabilities(
        supported={
            BrokerCapability.MARKET_ORDERS,
            BrokerCapability.LIMIT_ORDERS,
            BrokerCapability.POSITIONS,
            BrokerCapability.IDEMPOTENCY,
        }
    )

    assert caps.supports(BrokerCapability.MARKET_ORDERS) is True
    assert caps.supports("MARKET_ORDERS") is True
    assert caps.supports(BrokerCapability.EXTENDED_HOURS) is False

    # Enforcement passes for supported
    caps.require(BrokerCapability.MARKET_ORDERS)

    # Enforcement fails for unsupported
    with pytest.raises(BrokerUnsupportedOperationError, match="EXTENDED_HOURS"):
        caps.require(BrokerCapability.EXTENDED_HOURS, provider_name="TestBroker")

    # Paper adapter capabilities check
    adapter = PaperBrokerAdapter()
    adapter_caps = adapter.get_capabilities()
    assert adapter_caps.supports(BrokerCapability.MARKET_ORDERS) is True
    assert adapter_caps.supports(BrokerCapability.IDEMPOTENCY) is True


# =========================================================================
# 4. Error Hierarchy & Classification
# =========================================================================

def test_broker_error_hierarchy_and_retryability():
    """Verify error taxonomy correctly separates retryable vs non-retryable failures."""
    # Retryable errors
    err_conn = BrokerConnectionError("Socket disconnected")
    err_timeout = BrokerTimeoutError("HTTP 504 gateway timeout")
    err_rate = BrokerRateLimitError("Rate limit exceeded 429", retry_after=5.0)

    assert is_broker_error_retryable(err_conn) is True
    assert is_broker_error_retryable(err_timeout) is True
    assert is_broker_error_retryable(err_rate) is True
    assert err_rate.details.get("retry_after") == 5.0

    # Non-retryable errors
    err_auth = BrokerAuthenticationError("Invalid API key")
    err_rej = BrokerOrderRejectedError("Insufficient funds", rejection_reason="NO_FUNDS")
    err_inv = BrokerInvalidRequestError("Missing symbol")
    err_unsupp = BrokerUnsupportedOperationError("Feature not implemented")

    assert is_broker_error_retryable(err_auth) is False
    assert is_broker_error_retryable(err_rej) is False
    assert is_broker_error_retryable(err_inv) is False
    assert is_broker_error_retryable(err_unsupp) is False
    assert is_broker_error_retryable(ValueError("Generic error")) is False


# =========================================================================
# 5. Broker Factory & Registry Resolution
# =========================================================================

def test_broker_factory_paper_resolution():
    """Verify BrokerFactory creates PaperBrokerAdapter when configured for PAPER."""
    cfg = BrokerConfig(
        provider=BrokerProviderType.PAPER,
        execution_mode=BrokerExecutionMode.PAPER,
        initial_capital=75000.0,
    )
    broker = BrokerFactory.create_broker(config=cfg)
    assert isinstance(broker, BaseBroker)
    assert isinstance(broker, PaperBrokerAdapter)
    assert broker.get_account().cash == 75000.0


def test_broker_factory_live_mode_forbidden_for_paper_provider():
    """Verify BrokerFactory strictly forbids LIVE execution mode for Paper provider."""
    cfg = BrokerConfig(
        provider=BrokerProviderType.PAPER,
        execution_mode=BrokerExecutionMode.LIVE,
    )
    with pytest.raises(UnsupportedBrokerError, match="PAPER broker provider cannot run in LIVE execution mode"):
        BrokerFactory.create_broker(config=cfg)


def test_broker_factory_unsupported_provider_fails_closed():
    """Verify BrokerFactory fails closed with clear error for unsupported providers."""
    cfg = BrokerConfig(
        provider=BrokerProviderType.INTERACTIVE_BROKERS,
        execution_mode=BrokerExecutionMode.PAPER,
    )
    with pytest.raises(UnsupportedBrokerError, match="INTERACTIVE_BROKERS.*not supported"):
        BrokerFactory.create_broker(config=cfg)


# =========================================================================
# 6. Broker Health Check
# =========================================================================

def test_broker_health_check():
    """Verify health check returns valid BrokerHealthSnapshot."""
    adapter = PaperBrokerAdapter()
    health = adapter.health_check()
    assert isinstance(health, BrokerHealthSnapshot)
    assert health.status == BrokerStatus.CONNECTED
    assert health.connected is True
    assert health.provider == "PAPER"
    assert health.execution_mode == "PAPER"
    assert adapter.get_status() == BrokerStatus.CONNECTED


# =========================================================================
# 7. Live Safety Boundary Tests
# =========================================================================

def test_live_safety_boundary_no_credentials_required():
    """Verify broker abstraction operates with zero API keys or live credentials."""
    adapter = PaperBrokerAdapter()
    assert not hasattr(adapter, "api_key")
    assert not hasattr(adapter, "api_secret")
    assert not hasattr(adapter, "oauth_token")
    assert not hasattr(adapter, "base_live_url")


# =========================================================================
# 8. Determinism & Empirical Replay Verification
# =========================================================================

def test_broker_adapter_determinism():
    """Verify identical order executions through PaperBrokerAdapter yield identical account states."""
    t0 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    def run_simulation():
        adapter = PaperBrokerAdapter(config=BrokerConfig(initial_capital=100000.0))
        req = OrderRequest(
            client_order_id="det_ord_1",
            symbol="AAPL",
            side=BrokerSide.BUY,
            quantity=100.0,
            submitted_at=t0,
        )
        ord_rec = adapter.submit_order(req)
        adapter.execute_market_order(ord_rec.broker_order_id, market_price=150.0, timestamp=t0)
        return adapter.get_account(), adapter.get_positions()

    acc1, pos1 = run_simulation()
    acc2, pos2 = run_simulation()

    assert acc1.cash == acc2.cash
    assert acc1.equity == acc2.equity
    assert pos1[0].quantity == pos2[0].quantity
    assert pos1[0].average_price == pos2[0].average_price


def test_autonomous_controller_with_broker_adapter_historical_replay():
    """Verify AutonomousTradingController executes full historical replay via BaseBroker adapter."""
    storage = ProcessedDataStorage()
    aapl_bars = storage.load_bars("AAPL", TimeFrame.DAY_1, format="parquet")
    assert len(aapl_bars) >= 500

    ctrl = AutonomousTradingController(
        config=AutonomousConfig(
            universe=["AAPL"],
            paper_initial_capital=100000.0,
        )
    )
    # Verify broker interface is attached
    assert isinstance(ctrl.broker, BaseBroker)
    assert isinstance(ctrl.broker, PaperBrokerAdapter)

    ctrl.start()
    for bar in aapl_bars:
        ctrl.execute_market_event(market_timestamp=bar.timestamp, current_bars={"AAPL": bar})
    ctrl.stop()

    res = ctrl.get_result()
    assert res.session.total_cycles == len(aapl_bars)
    assert res.session.successful_cycles == len(aapl_bars)
    assert res.session.failed_cycles == 0

    # Verify broker reflects identical final account
    broker_acc = ctrl.broker.get_account()
    assert broker_acc.cash > 0.0
    assert broker_acc.equity > 0.0


# =========================================================================
# 9. Hardening: Order State Machine Transitions
# =========================================================================

def test_order_status_state_machine_transitions():
    """Verify legal vs illegal order lifecycle state transitions."""
    # 1. Legal forward paths
    assert validate_order_status_transition(BrokerOrderStatus.CREATED, BrokerOrderStatus.SUBMITTED) is True
    assert validate_order_status_transition(BrokerOrderStatus.SUBMITTED, BrokerOrderStatus.ACCEPTED) is True
    assert validate_order_status_transition(BrokerOrderStatus.ACCEPTED, BrokerOrderStatus.PARTIALLY_FILLED) is True
    assert validate_order_status_transition(BrokerOrderStatus.PARTIALLY_FILLED, BrokerOrderStatus.FILLED) is True
    assert validate_order_status_transition(BrokerOrderStatus.ACCEPTED, BrokerOrderStatus.CANCEL_PENDING) is True
    assert validate_order_status_transition(BrokerOrderStatus.CANCEL_PENDING, BrokerOrderStatus.CANCELLED) is True
    assert validate_order_status_transition(BrokerOrderStatus.ACCEPTED, BrokerOrderStatus.EXPIRED) is True
    assert validate_order_status_transition(BrokerOrderStatus.SUBMITTED, BrokerOrderStatus.REJECTED) is True

    # 2. Same-state idempotency
    assert validate_order_status_transition(BrokerOrderStatus.FILLED, BrokerOrderStatus.FILLED) is True
    assert validate_order_status_transition(BrokerOrderStatus.ACCEPTED, BrokerOrderStatus.ACCEPTED) is True

    # 3. Illegal terminal transitions
    with pytest.raises(ValueError, match="Illegal order status transition"):
        validate_order_status_transition(BrokerOrderStatus.FILLED, BrokerOrderStatus.SUBMITTED)

    with pytest.raises(ValueError, match="Illegal order status transition"):
        validate_order_status_transition(BrokerOrderStatus.CANCELLED, BrokerOrderStatus.FILLED)

    with pytest.raises(ValueError, match="Illegal order status transition"):
        validate_order_status_transition(BrokerOrderStatus.REJECTED, BrokerOrderStatus.ACCEPTED)

    with pytest.raises(ValueError, match="Illegal order status transition"):
        validate_order_status_transition(BrokerOrderStatus.EXPIRED, BrokerOrderStatus.SUBMITTED)


# =========================================================================
# 10. Hardening: Ambiguous Submission & Conflicting Idempotent Rejection
# =========================================================================

def test_ambiguous_submission_and_conflicting_idempotent_rejection():
    """Verify that conflicting duplicate order requests with the same client_order_id are rejected."""
    adapter = PaperBrokerAdapter(config=BrokerConfig(initial_capital=100000.0))
    req1 = OrderRequest(
        client_order_id="ambiguous_cid_999",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=50.0,
    )
    ord1 = adapter.submit_order(req1)
    assert ord1.client_order_id == "ambiguous_cid_999"

    # Exact duplicate matches seamlessly (idempotent recovery)
    ord1_dup = adapter.submit_order(req1)
    assert ord1_dup.broker_order_id == ord1.broker_order_id

    # Conflicting quantity with same client_order_id raises BrokerInvalidRequestError
    req_conflict_qty = OrderRequest(
        client_order_id="ambiguous_cid_999",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=100.0,
    )
    with pytest.raises(BrokerInvalidRequestError, match="Conflicting duplicate order request"):
        adapter.submit_order(req_conflict_qty)

    # Conflicting side with same client_order_id raises BrokerInvalidRequestError
    req_conflict_side = OrderRequest(
        client_order_id="ambiguous_cid_999",
        symbol="AAPL",
        side=BrokerSide.SELL,
        quantity=50.0,
    )
    with pytest.raises(BrokerInvalidRequestError, match="Conflicting duplicate order request"):
        adapter.submit_order(req_conflict_side)

    # Conflicting symbol with same client_order_id raises BrokerInvalidRequestError
    req_conflict_sym = OrderRequest(
        client_order_id="ambiguous_cid_999",
        symbol="MSFT",
        side=BrokerSide.BUY,
        quantity=50.0,
    )
    with pytest.raises(BrokerInvalidRequestError, match="Conflicting duplicate order request"):
        adapter.submit_order(req_conflict_sym)


# =========================================================================
# 11. Hardening: Capability-Aware Validation Enforcement
# =========================================================================

def test_capability_aware_validation_enforcement():
    """Verify broker rejects operations requiring capabilities that are disabled."""
    # Adapter without fractional share support
    adapter = PaperBrokerAdapter()
    adapter._capabilities.supported.discard(BrokerCapability.FRACTIONAL_SHARES)

    req_fractional = OrderRequest(
        client_order_id="frac_req_001",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=10.5,
    )
    with pytest.raises(BrokerUnsupportedOperationError, match="FRACTIONAL_SHARES"):
        adapter.submit_order(req_fractional)


# =========================================================================
# 12. Hardening: UNKNOWN State Safety
# =========================================================================

def test_unknown_state_handling_conservative():
    """Verify BrokerOrderStatus.UNKNOWN is handled conservatively (not terminal, not active)."""
    unknown_status = BrokerOrderStatus.UNKNOWN
    assert unknown_status.is_terminal() is False
    assert unknown_status.is_active() is False

    # Comparison with UNKNOWN flags status discrepancy
    loc_ord = BrokerOrder(
        broker_order_id="ord_1",
        client_order_id="cli_1",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=10.0,
        status=BrokerOrderStatus.FILLED,
        filled_quantity=10.0,
    )
    brk_ord = BrokerOrder(
        broker_order_id="ord_1",
        client_order_id="cli_1",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=10.0,
        status=BrokerOrderStatus.UNKNOWN,
        filled_quantity=0.0,
    )
    recon = reconcile_broker_order(loc_ord, brk_ord)
    assert recon.is_matched is False
    assert recon.status_match is False
    assert any("Status mismatch" in d for d in recon.discrepancies)


# =========================================================================
# 13. Hardening: Broker Order & Account Reconciliation
# =========================================================================

def test_broker_order_and_account_reconciliation():
    """Verify reconciliation comparison between local state and broker state."""
    # 1. Matching order reconciliation
    o1 = BrokerOrder(
        broker_order_id="ord_100",
        client_order_id="cli_100",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=50.0,
        filled_quantity=50.0,
        average_fill_price=150.0,
        status=BrokerOrderStatus.FILLED,
    )
    recon_order_match = reconcile_broker_order(o1, o1)
    assert recon_order_match.is_matched is True
    assert len(recon_order_match.discrepancies) == 0

    # 2. Mismatched order reconciliation
    o2_mismatched = BrokerOrder(
        broker_order_id="ord_100",
        client_order_id="cli_100",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=50.0,
        filled_quantity=25.0,  # Only partially filled on broker
        average_fill_price=149.0,
        status=BrokerOrderStatus.PARTIALLY_FILLED,
    )
    recon_order_mismatch = reconcile_broker_order(o1, o2_mismatched)
    assert recon_order_mismatch.is_matched is False
    assert recon_order_mismatch.status_match is False
    assert recon_order_mismatch.quantity_match is False

    # 3. Matching account reconciliation
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    acc1 = BrokerAccount(account_id="acc_1", cash=50000.0, equity=50000.0, buying_power=50000.0, timestamp=t0)
    acc2 = BrokerAccount(account_id="acc_1", cash=50000.0, equity=50000.0, buying_power=50000.0, timestamp=t0)
    recon_acc_match = reconcile_broker_account(acc1, acc2)
    assert recon_acc_match.is_matched is True
    assert recon_acc_match.cash_diff == 0.0

    # 4. Mismatched account reconciliation (e.g. broker cash drift)
    acc2_drift = BrokerAccount(account_id="acc_1", cash=49500.0, equity=49500.0, buying_power=49500.0, timestamp=t0)
    recon_acc_mismatch = reconcile_broker_account(acc1, acc2_drift)
    assert recon_acc_mismatch.is_matched is False
    assert recon_acc_mismatch.cash_diff == 500.0
    assert any("Cash mismatch" in d for d in recon_acc_mismatch.discrepancies)


# =========================================================================
# 14. Hardening: Timezone-Aware Timestamp Contract
# =========================================================================

def test_timezone_aware_timestamp_contract():
    """Verify all timestamps are timezone-aware and normalized to UTC."""
    t_naive = datetime(2025, 1, 1, 12, 0, 0)
    req = OrderRequest(
        client_order_id="ts_req_1",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=10.0,
        submitted_at=t_naive,
    )
    assert req.submitted_at.tzinfo is not None
    assert req.submitted_at.tzinfo == timezone.utc

    # Broker Execution timestamp normalization
    exec_rec = BrokerExecution(
        execution_id="exec_1",
        broker_order_id="ord_1",
        client_order_id="cli_1",
        symbol="AAPL",
        side=BrokerSide.BUY,
        quantity=10.0,
        execution_price=150.0,
        executed_at=t_naive,
    )
    assert exec_rec.executed_at.tzinfo is not None
    assert exec_rec.executed_at.tzinfo == timezone.utc

