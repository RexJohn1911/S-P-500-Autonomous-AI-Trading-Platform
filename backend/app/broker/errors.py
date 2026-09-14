"""
Normalized Broker Exceptions & Classification (Phase 16).
Provides strongly typed broker errors with explicit retry classification.
"""

from typing import Any, Dict, Optional


class BrokerError(Exception):
    """Base exception for all broker operations."""

    def __init__(
        self,
        message: str,
        provider: str = "PAPER",
        raw_error: Optional[Any] = None,
        status_code: Optional[int] = None,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.raw_error = raw_error
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "provider": self.provider,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "details": self.details,
        }


class BrokerConnectionError(BrokerError):
    """Transient network or socket connection failure."""

    def __init__(self, message: str, provider: str = "PAPER", **kwargs):
        super().__init__(message, provider=provider, retryable=True, **kwargs)


class BrokerTimeoutError(BrokerError):
    """Operation timed out waiting for broker response."""

    def __init__(self, message: str, provider: str = "PAPER", **kwargs):
        super().__init__(message, provider=provider, retryable=True, **kwargs)


class BrokerRateLimitError(BrokerError):
    """Broker API rate limit reached."""

    def __init__(self, message: str, provider: str = "PAPER", retry_after: Optional[float] = None, **kwargs):
        details = kwargs.pop("details", {})
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(message, provider=provider, retryable=True, details=details, **kwargs)


class BrokerAuthenticationError(BrokerError):
    """Invalid credentials, expired token, or unauthorized access."""

    def __init__(self, message: str, provider: str = "PAPER", **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class BrokerOrderRejectedError(BrokerError):
    """Order rejected by broker risk check, exchange, or validation."""

    def __init__(self, message: str, provider: str = "PAPER", rejection_reason: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if rejection_reason is not None:
            details["rejection_reason"] = rejection_reason
        self.rejection_reason = rejection_reason
        super().__init__(message, provider=provider, retryable=False, details=details, **kwargs)


class BrokerUnsupportedOperationError(BrokerError):
    """Operation or feature not supported by this broker provider or execution mode."""

    def __init__(self, message: str, provider: str = "PAPER", **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class UnsupportedBrokerError(BrokerUnsupportedOperationError):
    """Requested broker provider or execution mode is not supported."""
    pass


class BrokerInvalidRequestError(BrokerError):
    """Request payload validation failed or malformed arguments."""

    def __init__(self, message: str, provider: str = "PAPER", **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class BrokerAccountError(BrokerError):
    """Account restriction, margin call, or accounting error."""

    def __init__(self, message: str, provider: str = "PAPER", **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


class BrokerUnknownError(BrokerError):
    """Unclassified or unexpected broker exception."""

    def __init__(self, message: str, provider: str = "PAPER", **kwargs):
        super().__init__(message, provider=provider, retryable=False, **kwargs)


def is_broker_error_retryable(error: Exception) -> bool:
    """Return True if an exception is classified as a retryable broker error."""
    if isinstance(error, BrokerError):
        return error.retryable
    return False
