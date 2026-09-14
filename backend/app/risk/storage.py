"""
Risk Engine Storage Manager (Phase 12).
Handles atomic persistence and retrieval of risk engine assessments, adjusted targets,
and configuration artifacts under models/risk/{version}/.
"""

import json
from pathlib import Path
from typing import Optional, Union

from backend.app.risk.schemas import RiskEngineConfig, RiskEngineResult


class RiskStorage:
    """
    Manages filesystem serialization for risk engine artifacts.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir) if base_dir else Path("models/risk")

    def get_version_dir(self, version: str) -> Path:
        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir

    def save_result(
        self,
        result: RiskEngineResult,
        filepath: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save a RiskEngineResult instance to JSON."""
        if filepath is None:
            version_dir = self.get_version_dir(result.risk_version)
            ts_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
            target_path = version_dir / f"risk_result_{ts_str}.json"
        else:
            target_path = Path(filepath)
            target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return target_path

    def load_result(self, filepath: Union[str, Path]) -> RiskEngineResult:
        """Load a RiskEngineResult instance from JSON."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Risk result file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return RiskEngineResult.from_dict(data)

    def save_config(
        self,
        config: RiskEngineConfig,
        filepath: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save a RiskEngineConfig instance to JSON."""
        if filepath is None:
            version_dir = self.get_version_dir(config.version)
            target_path = version_dir / "risk_config.json"
        else:
            target_path = Path(filepath)
            target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

        return target_path

    def load_config(self, filepath: Union[str, Path]) -> RiskEngineConfig:
        """Load a RiskEngineConfig instance from JSON."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Risk config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return RiskEngineConfig.from_dict(data)
