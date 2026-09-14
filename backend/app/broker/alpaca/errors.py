"""
Alpaca Error Normalization (Phase 17).
Maps provider-specific HTTP codes and error payloads into normalized BrokerError types.
"""

import json
import logging
from typing import Any, Dict, Optional

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
)

logger = logging.getLogger(__name__)


def map_alpaca_error(
    status_code: int,
    response_text: str,
    operation: str = "request",
    details: Optional[Dict[str, Any]] = None,
) -> BrokerError:
    """
    Map an Alpaca HTTP response status and body to a normalized BrokerError.
    """
    err_details = details or {}
    message = f"Alpaca {operation} failed with status {status_code}"
    rejection_code = None

    try:
        if response_text:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                msg_val = parsed.get("message") or parsed.get("error") or parsed.get("description")
                if msg_val:
                    message = f"Alpaca {operation}: {msg_val}"
                rejection_code = parsed.get("code")
                err_details["alpaca_code"] = rejection_code
                err_details["raw_message"] = msg_val
    except Exception:
        err_details["raw_response"] = response_text[:200]

    if status_code == 401:
        return BrokerAuthenticationError(
            message=f"Alpaca authentication failed: Invalid or missing API credentials ({message})",
            provider="ALPACA",
            status_code=401,
            details=err_details,
        )

    if status_code == 403:
        return BrokerOrderRejectedError(
            message=f"Alpaca operation forbidden or restricted: {message}",
            provider="ALPACA",
            status_code=403,
            rejection_reason=str(rejection_code or "FORBIDDEN"),
            details=err_details,
        )

    if status_code == 404:
        return BrokerInvalidRequestError(
            message=f"Alpaca resource not found: {message}",
            provider="ALPACA",
            status_code=404,
            details=err_details,
        )

    if status_code == 422:
        return BrokerInvalidRequestError(
            message=f"Alpaca unprocessable order request: {message}",
            provider="ALPACA",
            status_code=422,
            details=err_details,
        )

    if status_code == 429:
        return BrokerRateLimitError(
            message=f"Alpaca rate limit exceeded: {message}",
            provider="ALPACA",
            status_code=429,
            retry_after=60.0,
            details=err_details,
        )

    if status_code in (500, 502, 503):
        return BrokerConnectionError(
            message=f"Alpaca server or gateway failure ({status_code}): {message}",
            provider="ALPACA",
            status_code=status_code,
            details=err_details,
        )

    if status_code == 504:
        return BrokerTimeoutError(
            message=f"Alpaca gateway timeout (504): {message}",
            provider="ALPACA",
            status_code=504,
            details=err_details,
        )

    return BrokerUnknownError(
        message=f"Alpaca unknown error ({status_code}): {message}",
        provider="ALPACA",
        status_code=status_code,
        details=err_details,
    )
