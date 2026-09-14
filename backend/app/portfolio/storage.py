"""
Portfolio Storage Manager.
Persists and retrieves Portfolio Construction configuration, targets, and execution-ready allocation results.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from backend.app.portfolio.schemas import (
    PortfolioConstructionConfig,
    PortfolioConstructionResult,
)

logger = logging.getLogger(__name__)


class PortfolioStorage:
    """
    Manages persistence of Portfolio Construction configurations and allocation results.
    Root path defaults to models/portfolio/.
    """

    def __init__(self, base_storage_dir: Optional[Union[str, Path]] = None):
        if base_storage_dir is not None:
            self.base_dir = Path(base_storage_dir)
        else:
            self.base_dir = Path("models/portfolio")

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
        config: PortfolioConstructionConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save PortfolioConstructionConfig to disk.
        """
        version_dir = self._get_version_dir(config.version)

        config_path = version_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

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
        meta_root = self.metadata_dir / f"portfolio_config_{config.version}.json"
        with open(meta_root, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        logger.info(f"PortfolioConstructionConfig [{config.version}] saved to {version_dir}")
        return version_dir

    def load_config(self, version: str = "portfolio-v1") -> PortfolioConstructionConfig:
        """
        Load PortfolioConstructionConfig from disk.
        """
        version_dir = self._get_version_dir(version)
        config_path = version_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Portfolio config artifact not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PortfolioConstructionConfig.from_dict(data)

    def save_result(
        self,
        result: PortfolioConstructionResult,
        version: str = "portfolio-v1",
        filename_prefix: str = "portfolio_result",
    ) -> Path:
        """
        Save PortfolioConstructionResult to disk.
        """
        version_dir = self._get_version_dir(version)
        result_path = version_dir / f"{filename_prefix}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"PortfolioConstructionResult saved to {result_path}")
        return result_path

    def load_result(
        self,
        version: str = "portfolio-v1",
        filename_prefix: str = "portfolio_result",
    ) -> PortfolioConstructionResult:
        """
        Load PortfolioConstructionResult from disk.
        """
        version_dir = self._get_version_dir(version)
        result_path = version_dir / f"{filename_prefix}.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Portfolio construction result not found at {result_path}")

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PortfolioConstructionResult.from_dict(data)
