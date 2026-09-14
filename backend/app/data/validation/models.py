"""
Data Quality & Validation Models
Defines structured results, severity tiers, issues, and aggregate quality reports.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from backend.app.data.models import TimeFrame, ensure_utc


class ValidationSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class DeduplicationPolicy(str, Enum):
    KEEP_LAST = "KEEP_LAST"
    KEEP_FIRST = "KEEP_FIRST"
    KEEP_HIGHER_VOLUME = "KEEP_HIGHER_VOLUME"
    REJECT_ON_DUPLICATE = "REJECT_ON_DUPLICATE"


class MarketSessionMode(str, Enum):
    REGULAR_HOURS = "REGULAR_HOURS"      # 09:30 - 16:00 America/New_York
    EXTENDED_HOURS = "EXTENDED_HOURS"    # 04:00 - 20:00 America/New_York
    ALL_HOURS = "ALL_HOURS"              # 24/7 (includes crypto/forex or full day)


@dataclass(frozen=True)
class ValidationIssue:
    """Individual data quality issue or observation."""
    validator_name: str
    severity: ValidationSeverity
    message: str
    symbol: Optional[str] = None
    timeframe: Optional[TimeFrame] = None
    timestamp: Optional[datetime] = None
    record_index: Optional[int] = None
    field_name: Optional[str] = None
    value_observed: Any = None
    expected: Optional[str] = None
    was_repaired: bool = False
    repair_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validator_name": self.validator_name,
            "severity": self.severity.value,
            "message": self.message,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value if self.timeframe else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "record_index": self.record_index,
            "field_name": self.field_name,
            "value_observed": str(self.value_observed) if self.value_observed is not None else None,
            "expected": self.expected,
            "was_repaired": self.was_repaired,
            "repair_action": self.repair_action,
        }


@dataclass
class ValidationResult:
    """Result of executing a specific validator over a dataset."""
    validator_name: str
    status: ValidationStatus
    records_inspected: int
    error_count: int = 0
    warning_count: int = 0
    repairs_performed: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    description: str = ""

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.error_count += 1
            self.status = ValidationStatus.FAIL
        elif issue.severity == ValidationSeverity.WARNING:
            self.warning_count += 1
            if self.status != ValidationStatus.FAIL:
                self.status = ValidationStatus.WARNING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validator_name": self.validator_name,
            "status": self.status.value,
            "records_inspected": self.records_inspected,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "repairs_performed": self.repairs_performed,
            "issues": [issue.to_dict() for issue in self.issues],
            "description": self.description,
        }


@dataclass
class DataQualityReport:
    """Aggregate quality report summarizing all validators for a dataset."""
    symbol: str
    timeframe: TimeFrame
    total_records: int
    valid_records_count: int
    overall_status: ValidationStatus
    results: List[ValidationResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_errors(self) -> int:
        return sum(r.error_count for r in self.results)

    @property
    def total_warnings(self) -> int:
        return sum(r.warning_count for r in self.results)

    @property
    def total_repairs(self) -> int:
        return sum(r.repairs_performed for r in self.results)

    @property
    def is_valid(self) -> bool:
        return self.overall_status != ValidationStatus.FAIL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "total_records": self.total_records,
            "valid_records_count": self.valid_records_count,
            "overall_status": self.overall_status.value,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "total_repairs": self.total_repairs,
            "timestamp": self.timestamp.isoformat(),
            "results": [res.to_dict() for res in self.results],
            "metadata": self.metadata,
        }

    def summary_text(self) -> str:
        lines = [
            f"=== DATA QUALITY REPORT [{self.symbol} | {self.timeframe.value}] ===",
            f"Overall Status: {self.overall_status.value}",
            f"Records Inspected: {self.total_records} | Valid: {self.valid_records_count}",
            f"Errors: {self.total_errors} | Warnings: {self.total_warnings} | Repairs: {self.total_repairs}",
        ]
        for r in self.results:
            status_indicator = "✓" if r.status == ValidationStatus.PASS else ("!" if r.status == ValidationStatus.WARNING else "✗")
            lines.append(f"  [{status_indicator}] {r.validator_name}: {r.status.value} (Errors: {r.error_count}, Warnings: {r.warning_count})")
        return "\n".join(lines)
