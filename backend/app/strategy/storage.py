"""
Signal Engine Storage Manager.
Persists and retrieves SignalEngine configuration, evaluation reports, and metadata.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from backend.app.strategy.schemas import SignalCandidate, SignalEngineConfig, SignalEvaluationReport

logger = logging.getLogger(__name__)


class SignalStorage:
    """
    Manages persistence of Signal Engine configurations, metadata, and evaluation diagnostics.
    Root path defaults to models/signals/.
    """

    def __init__(self, base_storage_dir: Optional[Union[str, Path]] = None):
        if base_storage_dir is not None:
            self.base_dir = Path(base_storage_dir)
        else:
            self.base_dir = Path("models/signals")

        self.metadata_dir = self.base_dir / "metadata"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create storage directories if they do not exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_version_dir(self, version: str) -> Path:
        target = self.base_dir / version
        target.mkdir(parents=True, exist_ok=True)
        return target

    def save_config(
        self,
        config: SignalEngineConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save SignalEngineConfig and metadata to disk.
        """
        version_dir = self._get_version_dir(config.version)

        # 1. Save config JSON
        config_path = version_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

        # 2. Save metadata JSON
        meta_payload = {
            "version": config.version,
            "saved_at": datetime.now().astimezone().isoformat(),
            "config": config.to_dict(),
            "custom_metadata": metadata or {},
        }
        meta_path = version_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        # Copy to root metadata directory
        meta_root = self.metadata_dir / f"signal_config_{config.version}.json"
        with open(meta_root, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        logger.info(f"SignalEngineConfig [{config.version}] saved successfully to {version_dir}")
        return version_dir

    def load_config(self, version: str = "signal-v1") -> SignalEngineConfig:
        """
        Load SignalEngineConfig from disk.
        """
        version_dir = self._get_version_dir(version)
        config_path = version_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Signal configuration artifact not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SignalEngineConfig.from_dict(data)

    def save_evaluation_report(
        self,
        report: SignalEvaluationReport,
        version: str = "signal-v1",
        filename_prefix: str = "eval_report",
    ) -> Path:
        """
        Save SignalEvaluationReport to disk.
        """
        version_dir = self._get_version_dir(version)
        report_path = version_dir / f"{filename_prefix}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"SignalEvaluationReport saved successfully to {report_path}")
        return report_path

    def load_evaluation_report(
        self,
        version: str = "signal-v1",
        filename_prefix: str = "eval_report",
    ) -> SignalEvaluationReport:
        """
        Load SignalEvaluationReport from disk.
        """
        version_dir = self._get_version_dir(version)
        report_path = version_dir / f"{filename_prefix}.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Signal evaluation report not found at {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SignalEvaluationReport.from_dict(data)
