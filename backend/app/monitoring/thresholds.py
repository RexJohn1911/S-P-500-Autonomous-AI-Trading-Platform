"""
Monitoring & Safety Threshold Configuration (Phase 18).
Provides validated thresholds and operational bounds.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MonitoringConfig:
    """Strongly-typed monitoring configuration with validated operational bounds."""
    enabled: bool = True
    heartbeat_timeout_seconds: float = 60.0
    max_cycle_duration_seconds: float = 120.0
    max_consecutive_failures: int = 3
    max_broker_latency_ms: float = 5000.0
    max_data_staleness_seconds: float = 86400.0 * 5
    max_reconciliation_age_seconds: float = 300.0
    alert_dedup_window_seconds: float = 60.0
    auto_kill_switch_enabled: bool = True
    storage_dir: str = "models/monitoring"
    version: str = "monitoring-v1.0"

    def validate(self) -> None:
        """Validate operational thresholds, raising ValueError if invalid."""
        if self.heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat_timeout_seconds must be positive")
        if self.max_cycle_duration_seconds <= 0:
            raise ValueError("max_cycle_duration_seconds must be positive")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be at least 1")
        if self.max_broker_latency_ms <= 0:
            raise ValueError("max_broker_latency_ms must be positive")
        if self.max_data_staleness_seconds < 0:
            raise ValueError("max_data_staleness_seconds must be non-negative")
        if self.max_reconciliation_age_seconds <= 0:
            raise ValueError("max_reconciliation_age_seconds must be positive")
        if self.alert_dedup_window_seconds < 0:
            raise ValueError("alert_dedup_window_seconds must be non-negative")
