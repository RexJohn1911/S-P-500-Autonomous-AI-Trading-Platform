"""
Base Validator Interface
Defines the abstract protocol for all modular data quality validators.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from backend.app.data.models import BarData, TimeFrame
from backend.app.data.validation.models import ValidationResult, ValidationStatus


class BaseValidator(ABC):
    """Abstract base class for all single-purpose data quality validators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this validator."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of what this validator checks."""
        pass

    @abstractmethod
    def validate(
        self,
        bars: List[BarData],
        expected_symbol: Optional[str] = None,
        expected_timeframe: Optional[TimeFrame] = None,
    ) -> ValidationResult:
        """
        Execute validation logic over a sequence of BarData records.
        Returns a structured ValidationResult.
        """
        pass
