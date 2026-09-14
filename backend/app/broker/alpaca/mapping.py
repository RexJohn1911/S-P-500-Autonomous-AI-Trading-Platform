"""
Alpaca Normalization Mapper (Phase 17).
Translates Alpaca REST payloads to and from normalized BaseBroker domain models.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional

from backend.app.broker.schemas import (
    BrokerAccount,
    BrokerExecution,
    BrokerOrder,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerSide,
    BrokerTimeInForce,
    OrderRequest,
)

logger = logging.getLogger(__name__)


def parse_alpaca_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse Alpaca ISO8601 timestamp string into timezone-aware UTC datetime."""
    if not ts_str:
        return None
    try:
        # Handle nanoseconds or trailing Z
        clean_ts = ts_str.rstrip("Z")
        if "." in clean_ts:
            parts = clean_ts.split(".")
            # Keep up to 6 decimal places for microseconds
            micro = parts[1][:6].ljust(6, "0")
            clean_ts = f"{parts[0]}.{micro}"
        dt = datetime.fromisoformat(clean_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as e:
        logger.warning("Failed to parse Alpaca timestamp '%s': %s", ts_str, e)
        return datetime.now(timezone.utc)


def alpaca_status_to_broker_status(status_str: Optional[str]) -> BrokerOrderStatus:
    """Map Alpaca order status string to normalized BrokerOrderStatus."""
    if not status_str:
        return BrokerOrderStatus.UNKNOWN

    s = status_str.lower().strip()
    mapping = {
        "new": BrokerOrderStatus.ACCEPTED,
        "pending_new": BrokerOrderStatus.ACCEPTED,
        "accepted": BrokerOrderStatus.ACCEPTED,
        "partially_filled": BrokerOrderStatus.PARTIALLY_FILLED,
        "filled": BrokerOrderStatus.FILLED,
        "done_for_day": BrokerOrderStatus.CANCELLED,
        "canceled": BrokerOrderStatus.CANCELLED,
        "cancelled": BrokerOrderStatus.CANCELLED,
        "pending_cancel": BrokerOrderStatus.CANCEL_PENDING,
        "expired": BrokerOrderStatus.EXPIRED,
        "replaced": BrokerOrderStatus.ACCEPTED,
        "pending_replace": BrokerOrderStatus.ACCEPTED,
        "accepted_for_bidding": BrokerOrderStatus.ACCEPTED,
        "stopped": BrokerOrderStatus.REJECTED,
        "rejected": BrokerOrderStatus.REJECTED,
        "suspended": BrokerOrderStatus.REJECTED,
        "calculated": BrokerOrderStatus.ACCEPTED,
    }
    return mapping.get(s, BrokerOrderStatus.UNKNOWN)


def alpaca_side_to_broker_side(side_str: str) -> BrokerSide:
    """Map Alpaca side string to BrokerSide."""
    return BrokerSide.BUY if side_str.lower() == "buy" else BrokerSide.SELL


def alpaca_type_to_broker_type(type_str: str) -> BrokerOrderType:
    """Map Alpaca order type string to BrokerOrderType."""
    t = type_str.lower()
    mapping = {
        "market": BrokerOrderType.MARKET,
        "limit": BrokerOrderType.LIMIT,
        "stop": BrokerOrderType.STOP,
        "stop_limit": BrokerOrderType.STOP_LIMIT,
    }
    return mapping.get(t, BrokerOrderType.MARKET)


def alpaca_tif_to_broker_tif(tif_str: str) -> BrokerTimeInForce:
    """Map Alpaca time in force string to BrokerTimeInForce."""
    tif = tif_str.lower()
    mapping = {
        "day": BrokerTimeInForce.DAY,
        "gtc": BrokerTimeInForce.GTC,
        "ioc": BrokerTimeInForce.IOC,
        "fok": BrokerTimeInForce.FOK,
    }
    return mapping.get(tif, BrokerTimeInForce.DAY)


def order_request_to_alpaca_payload(req: OrderRequest) -> Dict[str, Any]:
    """Convert normalized OrderRequest into Alpaca POST /v2/orders payload."""
    payload: Dict[str, Any] = {
        "symbol": req.symbol,
        "qty": float(req.quantity),
        "side": req.side.value.lower(),
        "type": req.order_type.value.lower().replace("_", ""),  # 'stop_limit' -> 'stoplimit' if needed, or 'stop_limit'
        "time_in_force": req.time_in_force.value.lower(),
        "client_order_id": req.client_order_id,
    }
    if req.order_type == BrokerOrderType.LIMIT:
        payload["type"] = "limit"
        payload["limit_price"] = str(req.limit_price)
    elif req.order_type == BrokerOrderType.STOP:
        payload["type"] = "stop"
        payload["stop_price"] = str(req.stop_price)
    elif req.order_type == BrokerOrderType.STOP_LIMIT:
        payload["type"] = "stop_limit"
        payload["limit_price"] = str(req.limit_price)
        payload["stop_price"] = str(req.stop_price)
    else:
        payload["type"] = "market"

    return payload


def alpaca_account_to_broker_account(data: Dict[str, Any]) -> BrokerAccount:
    """Normalize Alpaca GET /v2/account payload to BrokerAccount."""
    account_id = str(data.get("id") or data.get("account_number") or "alpaca_acc")
    currency = str(data.get("currency") or "USD")
    cash = float(data.get("cash") or 0.0)
    equity = float(data.get("equity") or data.get("portfolio_value") or cash)
    buying_power = float(data.get("buying_power") or data.get("regt_buying_power") or cash)
    available_cash = float(data.get("cash") or 0.0)
    status = str(data.get("status") or "ACTIVE").upper()

    return BrokerAccount(
        account_id=account_id,
        currency=currency,
        cash=cash,
        buying_power=buying_power,
        equity=equity,
        available_cash=available_cash,
        margin_used=float(data.get("initial_margin") or 0.0),
        margin_available=float(data.get("margin_available") or buying_power),
        status=status,
        timestamp=datetime.now(timezone.utc),
        metadata={
            "pattern_day_trader": data.get("pattern_day_trader", False),
            "trading_blocked": data.get("trading_blocked", False),
            "transfers_blocked": data.get("transfers_blocked", False),
            "account_blocked": data.get("account_blocked", False),
            "shorting_enabled": data.get("shorting_enabled", True),
        },
    )


def alpaca_position_to_broker_position(data: Dict[str, Any]) -> BrokerPosition:
    """Normalize Alpaca GET /v2/positions payload to BrokerPosition."""
    symbol = str(data.get("symbol")).upper()
    qty = float(data.get("qty") or 0.0)
    side_str = str(data.get("side") or ("long" if qty >= 0 else "short")).lower()
    if side_str == "short" and qty > 0:
        qty = -qty

    avg_entry_price = float(data.get("avg_entry_price") or 0.0)
    current_price = float(data.get("current_price") or avg_entry_price)
    market_value = float(data.get("market_value") or (qty * current_price))
    unrealized_pl = float(data.get("unrealized_pl") or 0.0)
    unrealized_intraday_pl = float(data.get("unrealized_intraday_pl") or 0.0)

    return BrokerPosition(
        symbol=symbol,
        quantity=qty,
        average_price=avg_entry_price,
        market_price=current_price,
        market_value=market_value,
        unrealized_pnl=unrealized_pl,
        realized_pnl=0.0,
        currency="USD",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "exchange": data.get("exchange"),
            "asset_class": data.get("asset_class"),
            "unrealized_intraday_pl": unrealized_intraday_pl,
            "change_today": float(data.get("change_today") or 0.0),
        },
    )


def alpaca_order_to_broker_order(data: Dict[str, Any]) -> BrokerOrder:
    """Normalize Alpaca GET /v2/orders payload to BrokerOrder."""
    broker_order_id = str(data.get("id"))
    client_order_id = str(data.get("client_order_id") or "")
    symbol = str(data.get("symbol")).upper()
    side = alpaca_side_to_broker_side(str(data.get("side", "buy")))
    qty = float(data.get("qty") or data.get("notional") or 0.0)
    filled_qty = float(data.get("filled_qty") or 0.0)
    remaining_qty = max(0.0, qty - filled_qty)

    order_type = alpaca_type_to_broker_type(str(data.get("type", "market")))
    limit_price = float(data["limit_price"]) if data.get("limit_price") is not None else None
    stop_price = float(data["stop_price"]) if data.get("stop_price") is not None else None
    tif = alpaca_tif_to_broker_tif(str(data.get("time_in_force", "day")))
    status = alpaca_status_to_broker_status(str(data.get("status", "unknown")))

    submitted_at = parse_alpaca_timestamp(data.get("submitted_at") or data.get("created_at"))
    accepted_at = parse_alpaca_timestamp(data.get("created_at"))
    filled_at = parse_alpaca_timestamp(data.get("filled_at"))
    cancelled_at = parse_alpaca_timestamp(data.get("canceled_at") or data.get("cancelled_at"))
    rejected_at = parse_alpaca_timestamp(data.get("rejected_at")) if status == BrokerOrderStatus.REJECTED else None

    avg_fill_price = float(data["filled_avg_price"]) if data.get("filled_avg_price") is not None else None

    return BrokerOrder(
        broker_order_id=broker_order_id,
        client_order_id=client_order_id,
        symbol=symbol,
        side=side,
        quantity=qty,
        filled_quantity=filled_qty,
        remaining_quantity=remaining_qty,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force=tif,
        status=status,
        submitted_at=submitted_at,
        accepted_at=accepted_at,
        filled_at=filled_at,
        cancelled_at=cancelled_at,
        rejected_at=rejected_at,
        rejection_reason=str(data.get("rejection_reason")) if data.get("rejection_reason") else None,
        average_fill_price=avg_fill_price,
        commission=0.0,
        currency="USD",
        metadata={
            "alpaca_status": data.get("status"),
            "extended_hours": data.get("extended_hours", False),
            "order_class": data.get("order_class"),
        },
    )


def alpaca_activity_to_broker_execution(data: Dict[str, Any]) -> BrokerExecution:
    """Normalize Alpaca GET /v2/account/activities payload to BrokerExecution."""
    exec_id = str(data.get("id") or data.get("transaction_time") or "exec")
    order_id = str(data.get("order_id") or "")
    symbol = str(data.get("symbol", "")).upper()
    side = alpaca_side_to_broker_side(str(data.get("side", "buy")))
    qty = float(data.get("qty") or 0.0)
    px = float(data.get("price") or 0.0)
    ts = parse_alpaca_timestamp(data.get("transaction_time") or data.get("date")) or datetime.now(timezone.utc)

    return BrokerExecution(
        execution_id=exec_id,
        broker_order_id=order_id,
        client_order_id=str(data.get("client_order_id") or ""),
        symbol=symbol,
        side=side,
        quantity=qty,
        execution_price=px,
        executed_at=ts,
        commission=0.0,
        currency="USD",
        venue="ALPACA",
        realized_pnl=None,
        metadata={
            "activity_type": data.get("activity_type"),
            "cum_qty": data.get("cum_qty"),
            "leaves_qty": data.get("leaves_qty"),
        },
    )
