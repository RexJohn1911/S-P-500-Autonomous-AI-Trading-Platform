"""
Alpaca Broker Schemas and Configuration (Phase 17).
Provides strongly typed configurations and Alpaca payload definitions with secret redaction.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class AlpacaEnvironment(str, Enum):
    """Alpaca environment targeting paper sandbox or live endpoint."""
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca Trading API Client."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    environment: AlpacaEnvironment = AlpacaEnvironment.PAPER
    base_url: Optional[str] = None
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_base: float = 1.0
    rate_limit_per_minute: int = 200

    def __post_init__(self):
        if self.base_url is None:
            if self.environment == AlpacaEnvironment.LIVE:
                self.base_url = "https://api.alpaca.markets"
            else:
                self.base_url = "https://paper-api.alpaca.markets"

    def is_configured(self) -> bool:
        """Return True if credentials are non-empty."""
        return bool(self.api_key and self.api_secret and len(self.api_key) > 3 and len(self.api_secret) > 3)

    def to_redacted_dict(self) -> Dict[str, Any]:
        """Return dict with secrets safely redacted for logging and inspection."""
        return {
            "environment": self.environment.value,
            "base_url": self.base_url,
            "api_key": "********" if self.api_key else None,
            "api_secret": "********" if self.api_secret else None,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "rate_limit_per_minute": self.rate_limit_per_minute,
        }
