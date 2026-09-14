"""
Backtest Storage Manager (Phase 13).
Handles serialization, storage, and retrieval of backtest results, configs, summaries,
equity curves, executions, and trade records under models/backtests/{backtest_id}/.
"""

import json
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from backend.app.backtest.schemas import BacktestConfig, BacktestResult, BacktestSummary


class BacktestStorage:
    """
    Manages filesystem serialization for backtest execution artifacts.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir) if base_dir else Path("models/backtests")

    def get_backtest_dir(self, backtest_id: str) -> Path:
        bt_dir = self.base_dir / backtest_id
        bt_dir.mkdir(parents=True, exist_ok=True)
        return bt_dir

    def save_result(
        self,
        result: BacktestResult,
        target_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Save complete BacktestResult artifacts into a dedicated directory.
        Creates:
        - config.json
        - summary.json
        - metrics.json
        - metadata.json
        - equity_curve.parquet / json
        - trades.parquet / json
        - executions.parquet / json
        - positions.parquet / json
        """
        out_dir = Path(target_dir) if target_dir else self.get_backtest_dir(result.backtest_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save config.json
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(result.config.to_dict(), f, indent=2)

        # 2. Save summary.json
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(result.summary.to_dict(), f, indent=2)

        # 3. Save metrics.json
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result.metrics.to_dict(), f, indent=2)

        # 4. Save metadata.json
        metadata = {
            "backtest_id": result.backtest_id,
            "evaluated_at": result.evaluated_at.isoformat(),
            "model_version": result.model_version,
            "signal_version": result.signal_version,
            "portfolio_version": result.portfolio_version,
            "risk_version": result.risk_version,
            "num_bars": len(result.equity_curve),
            "num_trades": len(result.trades),
            "num_executions": len(result.executions),
            "disclaimer": result.disclaimer,
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 5. Save Equity Curve
        if result.equity_curve:
            eq_dicts = [ep.to_dict() for ep in result.equity_curve]
            df_eq = pd.DataFrame(eq_dicts)
            try:
                df_eq.to_parquet(out_dir / "equity_curve.parquet", index=False)
            except Exception:
                pass
            with open(out_dir / "equity_curve.json", "w", encoding="utf-8") as f:
                json.dump(eq_dicts, f, indent=2)

        # 6. Save Trades
        if result.trades:
            trade_dicts = [t.to_dict() for t in result.trades]
            df_trades = pd.DataFrame(trade_dicts)
            try:
                df_trades.to_parquet(out_dir / "trades.parquet", index=False)
            except Exception:
                pass
            with open(out_dir / "trades.json", "w", encoding="utf-8") as f:
                json.dump(trade_dicts, f, indent=2)

        # 7. Save Executions
        if result.executions:
            exec_dicts = [ex.to_dict() for ex in result.executions]
            df_execs = pd.DataFrame(exec_dicts)
            try:
                df_execs.to_parquet(out_dir / "executions.parquet", index=False)
            except Exception:
                pass
            with open(out_dir / "executions.json", "w", encoding="utf-8") as f:
                json.dump(exec_dicts, f, indent=2)

        # 8. Save Positions (from latest portfolio state or history)
        if result.portfolio_history:
            pos_records = []
            for ps in result.portfolio_history:
                for sym, p in ps.positions.items():
                    p_dict = p.to_dict()
                    p_dict["timestamp"] = ps.timestamp.isoformat()
                    pos_records.append(p_dict)
            if pos_records:
                df_pos = pd.DataFrame(pos_records)
                try:
                    df_pos.to_parquet(out_dir / "positions.parquet", index=False)
                except Exception:
                    pass
                with open(out_dir / "positions.json", "w", encoding="utf-8") as f:
                    json.dump(pos_records, f, indent=2)

        # 9. Save complete result json
        with open(out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return out_dir

    def load_result(self, target_dir_or_file: Union[str, Path]) -> BacktestResult:
        """
        Load BacktestResult from directory or result.json.
        """
        path = Path(target_dir_or_file)
        if path.is_dir():
            result_file = path / "result.json"
        else:
            result_file = path

        if not result_file.exists():
            raise FileNotFoundError(f"Backtest result file not found: {result_file}")

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return BacktestResult.from_dict(data)
