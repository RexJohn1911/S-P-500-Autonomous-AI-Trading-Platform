"""
Broker Factory (Phase 16).
Instantiates standardized BaseBroker instances based on configuration.
Enforces strict fail-closed policy: LIVE execution is explicitly blocked in Phase 16.
"""

import logging
from typing import Optional

from backend.app.broker.alpaca.adapter import AlpacaBrokerAdapter
from backend.app.broker.alpaca.schemas import AlpacaConfig, AlpacaEnvironment
from backend.app.broker.errors import UnsupportedBrokerError
from backend.app.broker.interface import BaseBroker
from backend.app.broker.paper import PaperBrokerAdapter
from backend.app.broker.registry import BrokerRegistry
from backend.app.broker.schemas import BrokerConfig, BrokerExecutionMode, BrokerProviderType
from backend.app.paper_trading.broker import SimulatedPaperBroker

logger = logging.getLogger(__name__)


class BrokerFactory:
    """
    Factory for instantiating broker implementations.
    """

    @staticmethod
    def create_broker(
        config: Optional[BrokerConfig] = None,
        simulated_broker: Optional[SimulatedPaperBroker] = None,
        alpaca_config: Optional[AlpacaConfig] = None,
        session_id: Optional[str] = None,
    ) -> BaseBroker:
        """
        Create and return a BaseBroker implementation according to config.
        """
        cfg = config or BrokerConfig()

        # Strict safety check: PAPER provider cannot run in LIVE execution mode
        if cfg.provider == BrokerProviderType.PAPER and cfg.execution_mode == BrokerExecutionMode.LIVE:
            raise UnsupportedBrokerError(
                "PAPER broker provider cannot run in LIVE execution mode. Fail-closed.",
                provider=cfg.provider.value,
            )

        if cfg.provider not in (BrokerProviderType.PAPER, BrokerProviderType.ALPACA):
            raise UnsupportedBrokerError(
                f"Broker provider '{cfg.provider.value}' is not supported. "
                f"Supported providers: PAPER, ALPACA.",
                provider=cfg.provider.value,
            )

        broker_cls = BrokerRegistry.get(cfg.provider, cfg.execution_mode)

        # Instantiate paper adapter
        if broker_cls is PaperBrokerAdapter:
            return PaperBrokerAdapter(
                config=cfg,
                simulated_broker=simulated_broker,
                session_id=session_id,
            )

        # Instantiate Alpaca adapter
        if broker_cls is AlpacaBrokerAdapter:
            return AlpacaBrokerAdapter(
                config=cfg,
                alpaca_config=alpaca_config,
                session_id=session_id,
            )

        return broker_cls(config=cfg, session_id=session_id)
