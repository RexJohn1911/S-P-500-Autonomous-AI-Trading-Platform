"""
Operational Metrics Collector (Phase 18).
Collects real-time operational counters, latencies, and error statistics.
Observational only; does not affect trading decisions.
"""

from threading import RLock
from typing import Dict, Optional

from backend.app.monitoring.schemas import OperationalMetrics


class MetricsCollector:
    """
    Thread-safe operational metrics aggregator.
    """

    def __init__(self):
        self._lock = RLock()
        self._metrics = OperationalMetrics()
        self._total_cycle_time_ms: float = 0.0
        self._total_broker_latency_ms: float = 0.0
        self._broker_calls_count: int = 0

    def record_cycle(self, success: bool, duration_ms: float) -> None:
        with self._lock:
            self._metrics.cycles_total += 1
            if success:
                self._metrics.cycles_successful += 1
            else:
                self._metrics.cycles_failed += 1
            self._total_cycle_time_ms += duration_ms
            self._metrics.avg_cycle_duration_ms = self._total_cycle_time_ms / max(1, self._metrics.cycles_total)

    def record_order_submission(self) -> None:
        with self._lock:
            self._metrics.orders_submitted += 1

    def record_order_fill(self, is_partial: bool = False) -> None:
        with self._lock:
            if is_partial:
                self._metrics.partial_fills += 1
            else:
                self._metrics.orders_filled += 1

    def record_order_rejection(self) -> None:
        with self._lock:
            self._metrics.orders_rejected += 1

    def record_broker_call(self, latency_ms: float, is_error: bool = False, is_rate_limit: bool = False) -> None:
        with self._lock:
            self._broker_calls_count += 1
            self._total_broker_latency_ms += latency_ms
            self._metrics.broker_avg_latency_ms = self._total_broker_latency_ms / max(1, self._broker_calls_count)
            if is_error:
                self._metrics.broker_errors_total += 1
            if is_rate_limit:
                self._metrics.rate_limit_events += 1

    def record_reconciliation_mismatch(self) -> None:
        with self._lock:
            self._metrics.reconciliation_mismatches += 1

    def record_stale_data(self) -> None:
        with self._lock:
            self._metrics.stale_data_events += 1

    def record_model_failure(self) -> None:
        with self._lock:
            self._metrics.model_inference_failures += 1

    def record_safety_block(self) -> None:
        with self._lock:
            self._metrics.safety_gate_blocks += 1

    def record_kill_switch_trigger(self) -> None:
        with self._lock:
            self._metrics.kill_switch_activations += 1

    def update_incident_counts(self, active: int, critical: int) -> None:
        with self._lock:
            self._metrics.active_incidents_count = active
            self._metrics.critical_incidents_count = critical

    def get_metrics(self) -> OperationalMetrics:
        with self._lock:
            return OperationalMetrics(
                cycles_total=self._metrics.cycles_total,
                cycles_successful=self._metrics.cycles_successful,
                cycles_failed=self._metrics.cycles_failed,
                avg_cycle_duration_ms=round(self._metrics.avg_cycle_duration_ms, 2),
                orders_submitted=self._metrics.orders_submitted,
                orders_filled=self._metrics.orders_filled,
                orders_rejected=self._metrics.orders_rejected,
                partial_fills=self._metrics.partial_fills,
                broker_errors_total=self._metrics.broker_errors_total,
                broker_avg_latency_ms=round(self._metrics.broker_avg_latency_ms, 2),
                rate_limit_events=self._metrics.rate_limit_events,
                reconciliation_mismatches=self._metrics.reconciliation_mismatches,
                active_incidents_count=self._metrics.active_incidents_count,
                critical_incidents_count=self._metrics.critical_incidents_count,
                kill_switch_activations=self._metrics.kill_switch_activations,
                safety_gate_blocks=self._metrics.safety_gate_blocks,
                stale_data_events=self._metrics.stale_data_events,
                model_inference_failures=self._metrics.model_inference_failures,
            )
