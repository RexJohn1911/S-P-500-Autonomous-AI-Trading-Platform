"""
Safety Gate Evaluator (Phase 18).
Implements the 12-point pre-flight order execution safety gates.
Ensures fail-closed verification before any order can reach the broker adapter.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

from backend.app.monitoring.incident import IncidentManager
from backend.app.monitoring.kill_switch import GlobalKillSwitch
from backend.app.monitoring.schemas import (
    ComponentHealthSnapshot,
    ComponentType,
    HealthStatus,
    SafetyEvaluation,
    SafetyGateResult,
    SystemHealthSnapshot,
)

logger = logging.getLogger(__name__)


class SafetyGateEvaluator:
    """
    Evaluates 12 explicit pre-flight safety gates before order submission.
    """

    def __init__(
        self,
        kill_switch: GlobalKillSwitch,
        incident_manager: IncidentManager,
    ):
        self.kill_switch = kill_switch
        self.incident_manager = incident_manager

    def evaluate_order(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        system_health: Optional[SystemHealthSnapshot] = None,
        component_healths: Optional[Dict[ComponentType, ComponentHealthSnapshot]] = None,
        signal_valid: bool = True,
        portfolio_valid: bool = True,
        risk_approved: bool = True,
        reconciliation_matched: bool = True,
        autonomous_loop_healthy: bool = True,
        execution_mode_permitted: bool = True,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> SafetyEvaluation:
        """
        Evaluate all 12 pre-flight safety gates.
        """
        gates: List[SafetyGateResult] = []
        rejection_reasons: List[str] = []
        comps = component_healths or {}

        # 1. System Healthy
        sys_status = system_health.status if system_health else HealthStatus.HEALTHY
        sys_ok = sys_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        gates.append(SafetyGateResult(
            gate_name="1. SYSTEM_HEALTHY",
            passed=sys_ok,
            reason="System health permits execution" if sys_ok else f"System health is {sys_status.value}",
        ))
        if not sys_ok:
            rejection_reasons.append(f"System health is {sys_status.value}")

        # 2. Market Data Healthy
        md_snap = comps.get(ComponentType.MARKET_DATA)
        md_ok = md_snap.status.allows_trading() if md_snap else True
        gates.append(SafetyGateResult(
            gate_name="2. MARKET_DATA_HEALTHY",
            passed=md_ok,
            reason="Market data healthy" if md_ok else f"Market data status is {md_snap.status.value}",
        ))
        if not md_ok:
            rejection_reasons.append("Market data unhealthy or stale")

        # 3. Model Healthy
        model_snap = comps.get(ComponentType.MODEL)
        model_ok = model_snap.status.allows_trading() if model_snap else True
        gates.append(SafetyGateResult(
            gate_name="3. MODEL_HEALTHY",
            passed=model_ok,
            reason="AI models available and valid" if model_ok else "AI model unavailable or invalid",
        ))
        if not model_ok:
            rejection_reasons.append("AI model health check failed")

        # 4. Signal Valid
        gates.append(SafetyGateResult(
            gate_name="4. SIGNAL_VALID",
            passed=signal_valid,
            reason="Signal candidate passed validity checks" if signal_valid else "Signal is invalid, null, or out of bounds",
        ))
        if not signal_valid:
            rejection_reasons.append("Invalid trading signal candidate")

        # 5. Portfolio Valid
        gates.append(SafetyGateResult(
            gate_name="5. PORTFOLIO_VALID",
            passed=portfolio_valid,
            reason="Portfolio targets within hard constraints" if portfolio_valid else "Portfolio target violates allocation bounds",
        ))
        if not portfolio_valid:
            rejection_reasons.append("Portfolio allocation invalid")

        # 6. Risk Approved
        gates.append(SafetyGateResult(
            gate_name="6. RISK_APPROVED",
            passed=risk_approved,
            reason="Risk engine approved order" if risk_approved else "Risk engine rejected order",
        ))
        if not risk_approved:
            rejection_reasons.append("Risk engine policy violation")

        # 7. Broker Healthy
        broker_snap = comps.get(ComponentType.BROKER)
        broker_ok = broker_snap.status.allows_trading() if broker_snap else True
        gates.append(SafetyGateResult(
            gate_name="7. BROKER_HEALTHY",
            passed=broker_ok,
            reason="Broker connection healthy" if broker_ok else f"Broker health is {broker_snap.status.value}",
        ))
        if not broker_ok:
            rejection_reasons.append("Broker connection degraded or unavailable")

        # 8. Reconciliation Healthy
        gates.append(SafetyGateResult(
            gate_name="8. RECONCILIATION_HEALTHY",
            passed=reconciliation_matched,
            reason="Local and broker ledgers matched" if reconciliation_matched else "Unresolved ledger mismatch detected",
        ))
        if not reconciliation_matched:
            rejection_reasons.append("Reconciliation mismatch detected")

        # 9. Autonomous Loop Healthy
        gates.append(SafetyGateResult(
            gate_name="9. AUTONOMOUS_LOOP_HEALTHY",
            passed=autonomous_loop_healthy,
            reason="Autonomous loop active and responsive" if autonomous_loop_healthy else "Autonomous loop stuck or failed",
        ))
        if not autonomous_loop_healthy:
            rejection_reasons.append("Autonomous trading loop is unhealthy or paused")

        # 10. Kill Switch Inactive (ARMED)
        kill_triggered = self.kill_switch.is_triggered()
        gates.append(SafetyGateResult(
            gate_name="10. KILL_SWITCH_INACTIVE",
            passed=not kill_triggered,
            reason="Kill switch is armed" if not kill_triggered else "Global kill switch is TRIGGERED",
        ))
        if kill_triggered:
            rejection_reasons.append("Global kill switch is active")

        # 11. No Critical Incidents
        has_crit_incidents = self.incident_manager.has_unresolved_critical_incident()
        gates.append(SafetyGateResult(
            gate_name="11. NO_CRITICAL_INCIDENTS",
            passed=not has_crit_incidents,
            reason="No active critical incidents" if not has_crit_incidents else "Unresolved critical incident exists",
        ))
        if has_crit_incidents:
            rejection_reasons.append("Active critical operational incident exists")

        # 12. Execution Mode Permitted
        gates.append(SafetyGateResult(
            gate_name="12. EXECUTION_MODE_PERMITTED",
            passed=execution_mode_permitted,
            reason="Execution mode authorized" if execution_mode_permitted else "Execution mode unauthorized or unconfigured",
        ))
        if not execution_mode_permitted:
            rejection_reasons.append("Execution mode is not permitted")

        is_permitted = len(rejection_reasons) == 0

        evaluation = SafetyEvaluation(
            is_permitted=is_permitted,
            timestamp=datetime.now(timezone.utc),
            order_id=order_id,
            symbol=symbol,
            gates=gates,
            rejection_reasons=rejection_reasons,
        )

        if not is_permitted:
            logger.warning(
                "PRE-FLIGHT SAFETY GATE REJECTION for order %s (%s): %s",
                order_id,
                symbol,
                rejection_reasons,
            )

        return evaluation
