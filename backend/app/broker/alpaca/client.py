"""
Alpaca Trading REST API Client Wrapper (Phase 17).
Handles authenticated HTTP transport, bounded retries, rate-limiting, and error mapping.
Secret-safe: credentials are never logged or exposed.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional
import httpx

from backend.app.broker.alpaca.errors import map_alpaca_error
from backend.app.broker.alpaca.schemas import AlpacaConfig, AlpacaEnvironment
from backend.app.broker.errors import (
    BrokerAuthenticationError,
    BrokerConnectionError,
    BrokerError,
    BrokerRateLimitError,
    BrokerTimeoutError,
)

logger = logging.getLogger(__name__)


class AlpacaTradingClient:
    """
    Low-level authenticated HTTP client for Alpaca v2 Trading API.
    """

    def __init__(
        self,
        config: Optional[AlpacaConfig] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        self.config = config or AlpacaConfig()
        self.base_url = self.config.base_url.rstrip("/")
        self._http_client = http_client
        self._owns_client = http_client is None

    def _get_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=self.config.timeout_seconds,
            )
        return self._http_client

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.config.api_key and self.config.api_secret:
            headers["APCA-API-KEY-ID"] = self.config.api_key
            headers["APCA-API-SECRET-KEY"] = self.config.api_secret
        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        operation: str = "request",
        allow_retries: bool = True,
    ) -> Any:
        """Execute authenticated request with rate-limiting backoff."""
        url = f"{self.base_url}{path}"
        client = self._get_client()
        headers = self._get_headers()

        max_attempts = self.config.max_retries if allow_retries else 1
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_data,
                )

                if response.status_code in (200, 201, 204):
                    if response.status_code == 204 or not response.text:
                        return {}
                    return response.json()

                # Handle failure status
                err = map_alpaca_error(
                    status_code=response.status_code,
                    response_text=response.text,
                    operation=operation,
                )

                if err.retryable and attempt < max_attempts and allow_retries:
                    backoff = self.config.retry_backoff_base * (2 ** (attempt - 1))
                    logger.warning(
                        "Alpaca %s retryable error %s (attempt %d/%d). Backing off %.1fs",
                        operation,
                        type(err).__name__,
                        attempt,
                        max_attempts,
                        backoff,
                    )
                    time.sleep(backoff)
                    last_error = err
                    continue

                raise err

            except httpx.TimeoutException as e:
                err = BrokerTimeoutError(
                    f"Alpaca {operation} timed out after {self.config.timeout_seconds}s: {e}",
                    provider="ALPACA",
                    details={"path": path, "method": method},
                )
                if attempt < max_attempts and allow_retries:
                    backoff = self.config.retry_backoff_base * (2 ** (attempt - 1))
                    time.sleep(backoff)
                    last_error = err
                    continue
                raise err

            except httpx.RequestError as e:
                err = BrokerConnectionError(
                    f"Alpaca {operation} connection failed: {e}",
                    provider="ALPACA",
                    details={"path": path, "method": method},
                )
                if attempt < max_attempts and allow_retries:
                    backoff = self.config.retry_backoff_base * (2 ** (attempt - 1))
                    time.sleep(backoff)
                    last_error = err
                    continue
                raise err

        if last_error:
            raise last_error

    # =========================================================================
    # Endpoint Methods
    # =========================================================================

    def get_account(self) -> Dict[str, Any]:
        """GET /v2/account."""
        return self._request("GET", "/v2/account", operation="get_account")

    def get_positions(self) -> List[Dict[str, Any]]:
        """GET /v2/positions."""
        res = self._request("GET", "/v2/positions", operation="get_positions")
        return res if isinstance(res, list) else []

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """GET /v2/positions/{symbol}."""
        try:
            return self._request("GET", f"/v2/positions/{symbol.upper()}", operation="get_position")
        except BrokerError as e:
            if getattr(e, "status_code", None) == 404:
                return None
            raise

    def get_orders(self, status: Optional[str] = "all", limit: int = 500) -> List[Dict[str, Any]]:
        """GET /v2/orders."""
        params = {"status": status, "limit": limit, "direction": "desc"}
        res = self._request("GET", "/v2/orders", params=params, operation="get_orders")
        return res if isinstance(res, list) else []

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """GET /v2/orders/{order_id}."""
        try:
            return self._request("GET", f"/v2/orders/{order_id}", operation="get_order")
        except BrokerError as e:
            if getattr(e, "status_code", None) == 404:
                return None
            raise

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """GET /v2/orders:by_client_order_id?client_order_id=..."""
        try:
            return self._request(
                "GET",
                "/v2/orders:by_client_order_id",
                params={"client_order_id": client_order_id},
                operation="get_order_by_client_id",
            )
        except BrokerError as e:
            if getattr(e, "status_code", None) == 404:
                return None
            raise

    def submit_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /v2/orders. (Do NOT auto-retry order submission on timeout)."""
        return self._request(
            "POST",
            "/v2/orders",
            json_data=payload,
            operation="submit_order",
            allow_retries=False,  # Single shot to prevent duplicate order generation
        )

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """DELETE /v2/orders/{order_id}."""
        return self._request("DELETE", f"/v2/orders/{order_id}", operation="cancel_order")

    def get_activities(self, activity_types: str = "FILL") -> List[Dict[str, Any]]:
        """GET /v2/account/activities/{activity_type}."""
        res = self._request(
            "GET",
            f"/v2/account/activities/{activity_types}",
            operation="get_activities",
        )
        return res if isinstance(res, list) else []

    def get_clock(self) -> Dict[str, Any]:
        """GET /v2/clock."""
        return self._request("GET", "/v2/clock", operation="get_clock")

    def close(self) -> None:
        """Close underlying HTTP client."""
        if self._owns_client and self._http_client is not None:
            self._http_client.close()
            self._http_client = None
