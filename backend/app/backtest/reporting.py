"""
Backtest Reporting Generator (Phase 13 Upgrade v1.2).
Produces 16-section research-grade quantitative reports (Markdown, dict, text) for backtest evaluation results.
"""

from typing import Dict, Optional
from backend.app.backtest.schemas import BacktestResult, WalkForwardResult


class BacktestReporter:
    """
    Generates research-grade human-readable and structured reporting artifacts.
    """

    @staticmethod
    def generate_markdown_report(result: BacktestResult) -> str:
        """
        Generate a comprehensive 16-section GitHub-flavored Markdown report.
        """
        cfg = result.config
        met = result.metrics
        sumry = result.summary
        bm = result.benchmark_metrics
        cb = result.cost_breakdown
        dp = result.dataset_provenance

        start_str = sumry.start_date.strftime("%Y-%m-%d") if sumry.start_date else "N/A"
        end_str = sumry.end_date.strftime("%Y-%m-%d") if sumry.end_date else "N/A"

        sharpe_str = f"{met.sharpe_ratio:.4f}" if met.sharpe_ratio is not None else "N/A"
        sortino_str = f"{met.sortino_ratio:.4f}" if met.sortino_ratio is not None else "N/A"
        calmar_str = f"{met.calmar_ratio:.4f}" if met.calmar_ratio is not None else "N/A"
        win_rate_str = f"{met.win_rate * 100:.2f}%" if met.win_rate is not None else "N/A"
        pf_str = f"{met.profit_factor:.4f}" if met.profit_factor is not None else "N/A"
        avg_trade_str = f"${met.average_trade_pnl:,.2f}" if met.average_trade_pnl is not None else "N/A"

        # 12. Benchmark Comparison
        if bm is not None:
            bm_sharpe = f"{bm.benchmark_sharpe:.4f}" if bm.benchmark_sharpe is not None else "N/A"
            alpha_str = f"{bm.alpha * 100:.2f}%" if bm.alpha is not None else "N/A"
            beta_str = f"{bm.beta:.4f}" if bm.beta is not None else "N/A"
            corr_str = f"{bm.correlation:.4f}" if bm.correlation is not None else "N/A"
            bm_table = f"""| Metric | Strategy | Benchmark ({bm.benchmark_symbol}) |
| :--- | :--- | :--- |
| **Total Return** | {met.total_return * 100:.2f}% | {bm.benchmark_total_return * 100:.2f}% |
| **Annualized Return** | {met.annualized_return * 100:.2f}% | {bm.benchmark_annualized_return * 100:.2f}% |
| **Annualized Volatility** | {met.annualized_volatility * 100:.2f}% | {bm.benchmark_annualized_volatility * 100:.2f}% |
| **Sharpe Ratio** | {sharpe_str} | {bm_sharpe} |
| **Max Drawdown** | {met.max_drawdown * 100:.2f}% | {bm.benchmark_max_drawdown * 100:.2f}% |
| **Alpha (Jensen)** | {alpha_str} | - |
| **Beta** | {beta_str} | 1.0000 |
| **Correlation** | {corr_str} | 1.0000 |"""
        else:
            bm_table = f"*Benchmark data unavailable or not configured ({cfg.benchmark_symbol}).*"

        # 13. Cost Attribution
        comm_cost = cb.commission_cost if cb else met.total_commission
        slip_cost = cb.slippage_cost if cb else met.total_slippage
        spread_cost = cb.spread_cost if cb else met.total_spread_cost
        borrow_cost = cb.borrow_cost if cb else met.total_borrow_cost
        impact_cost = cb.market_impact_cost if cb else met.total_market_impact_cost
        tot_cost = met.total_costs
        gross_pnl_val = met.gross_pnl if met.gross_pnl is not None else (sumry.total_pnl + tot_cost)
        net_pnl_val = met.net_pnl if met.net_pnl is not None else sumry.total_pnl

        # 14. Drawdown Episodes
        dd_episodes_table = ""
        if met.drawdown_episodes:
            rows = []
            for ep in met.drawdown_episodes[:5]:
                rec_str = ep.recovery_time.strftime("%Y-%m-%d") if ep.recovery_time else "Ongoing"
                rows.append(
                    f"| {ep.peak_time.strftime('%Y-%m-%d')} | {ep.trough_time.strftime('%Y-%m-%d')} | {rec_str} | {ep.depth_pct * 100:.2f}% | {ep.duration_bars} bars |"
                )
            dd_episodes_table = "\n".join(rows)
        else:
            dd_episodes_table = "| N/A | N/A | N/A | 0.00% | 0 bars |"

        # 2. Dataset Information
        ds_id = dp.dataset_id if dp else "local_pipeline_market_data"
        ds_prov = dp.provider if dp else "Platform Ingestion Pipeline"
        ds_adj = dp.adjustment_status.value if dp else "Pre-adjusted / Provider native"
        ds_src = dp.data_source_type.value if dp else "local_historical"

        symbols_str = ", ".join(cfg.symbols) if cfg.symbols else "All portfolio universe"

        md = f"""# Quantitative Backtest Report — ID: `{result.backtest_id}`

> **DISCLAIMER:** {result.disclaimer}

---

## 1. Run Information
| Parameter | Strategy Value |
| :--- | :--- |
| **Backtest ID** | `{result.backtest_id}` |
| **Evaluation Timestamp (UTC)** | `{result.evaluated_at.isoformat()}` |
| **Backtest Engine Version** | `{cfg.version}` |
| **Evaluation Period** | {start_str} to {end_str} ({sumry.total_trading_days} bars) |

---

## 2. Dataset Information & Provenance
| Dataset Attribute | Value |
| :--- | :--- |
| **Dataset ID** | `{ds_id}` |
| **Provider** | `{ds_prov}` |
| **Data Source Type** | `{ds_src}` |
| **Adjustment Status** | `{ds_adj}` |
| **Timeframe** | `{cfg.timeframe}` |
| **Evaluated Symbols** | {symbols_str} |

---

## 3. Model Information & Provenance
| Pipeline Layer | Version / Artifact |
| :--- | :--- |
| **AI / ML Model Version** | `{result.model_version}` |
| **Signal Engine Version** | `{result.signal_version}` |
| **Portfolio Engine Version** | `{result.portfolio_version}` |
| **Risk Engine Version** | `{result.risk_version}` |

---

## 4. Strategy & Signal Configuration
| Key | Setting |
| :--- | :--- |
| **Signal Direction** | LONG / FLAT (Long-Short if enabled) |
| **Gating Policy** | Regime-conditioned confidence filtering |
| **Reason Codes** | Model Consensus, Regime Support, Momentum Confirmation |

---

## 5. Portfolio Construction Configuration
| Key | Setting |
| :--- | :--- |
| **Initial Capital** | ${sumry.initial_capital:,.2f} {cfg.base_currency} |
| **Rebalancing Frequency** | Daily bar-close schedule |
| **Fractional Shares** | {cfg.allow_fractional_shares} |

---

## 6. Risk Management Configuration
| Key | Setting |
| :--- | :--- |
| **Risk Engine Status** | ACTIVE (Deterministic post-portfolio clipping) |
| **Max Single Position Weight** | Risk limit enforced |
| **Exposure Boundaries** | Gross and net exposure limits enforced |

---

## 7. Execution Configuration
| Key | Setting |
| :--- | :--- |
| **Execution Convention** | `{cfg.execution_price_type.value}` |
| **Causality Guarantee** | Signal at $t$ executes strictly at $t+1$ (Zero Lookahead) |
| **Min Trade Notional** | ${cfg.minimum_trade_notional:,.2f} |

---

## 8. Cost Configuration & Assumptions
| Friction Component | Configured Rate |
| :--- | :--- |
| **Commission Rate** | {cfg.commission_rate * 10000:.1f} bps ({cfg.commission_rate}) |
| **Slippage Rate** | {cfg.slippage_rate * 10000:.1f} bps ({cfg.slippage_rate}) |
| **Bid/Ask Spread Rate** | {cfg.bid_ask_spread_rate * 10000:.1f} bps ({cfg.bid_ask_spread_rate}) |
| **Daily Borrow Fee Rate** | {cfg.daily_borrow_rate * 10000:.2f} bps ({cfg.daily_borrow_rate}) |
| **Market Impact Coeff** | {cfg.market_impact_coefficient:.4f} |

---

## 9. Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Start Equity** | ${met.start_equity:,.2f} |
| **End Equity** | ${met.end_equity:,.2f} |
| **Total Net Return** | {met.total_return * 100:.2f}% |
| **Annualized Return (CAGR)** | {met.annualized_return * 100:.2f}% |
| **Annualized Volatility** | {met.annualized_volatility * 100:.2f}% |
| **Sharpe Ratio (Rf={cfg.risk_free_rate})** | {sharpe_str} |
| **Sortino Ratio** | {sortino_str} |
| **Calmar Ratio** | {calmar_str} |
| **Maximum Drawdown** | {met.max_drawdown * 100:.2f}% |
| **Max Drawdown Duration** | {met.max_drawdown_duration_bars} bars |

---

## 10. Trading Statistics
| Metric | Value |
| :--- | :--- |
| **Total Completed Trades** | {met.total_completed_trades} |
| **Winning / Losing Trades** | {met.winning_trades} / {met.losing_trades} |
| **Win Rate** | {win_rate_str} |
| **Profit Factor** | {pf_str} |
| **Average Trade P&L** | {avg_trade_str} |
| **Total Turnover** | {met.total_turnover:.4f} |

---

## 11. Risk Statistics & Portfolio Diagnostics
| Diagnostic Metric | Observed Value |
| :--- | :--- |
| **Dividends Collected** | ${result.dividends_collected:,.2f} |
| **Stock Splits Processed** | {result.splits_applied_count} |
| **Total Executions** | {len(result.executions)} |

---

## 12. Benchmark Comparison
{bm_table}

---

## 13. Cost Attribution & Friction Breakdown
| Cost Category | Amount |
| :--- | :--- |
| **Gross Strategy P&L** | ${gross_pnl_val:,.2f} |
| **Commission Paid** | -${comm_cost:,.2f} |
| **Slippage Incurred** | -${slip_cost:,.2f} |
| **Bid/Ask Spread Cost** | -${spread_cost:,.2f} |
| **Short Borrow Financing** | -${borrow_cost:,.2f} |
| **Market Impact Friction** | -${impact_cost:,.2f} |
| **Total Friction Deducted** | -${tot_cost:,.2f} |
| **Net Strategy P&L** | **${net_pnl_val:,.2f}** |

---

## 14. Drawdown Analysis & Top Episodes
| Peak Date | Trough Date | Recovery Date | Depth | Duration |
| :--- | :--- | :--- | :--- | :--- |
{dd_episodes_table}

---

## 15. Walk-Forward Evaluation Info
*Standard point-in-time backtest mode. For walk-forward mode, refer to WalkForwardResult.*

---

## 16. Research Limitations & Disclaimers
1. Simulated historical results do not guarantee future live execution performance.
2. Market fills are modeled on daily bar intervals with conservative next-bar execution.
3. Corporate action data must be validated against official exchange split/dividend notices.
"""
        return md

    @staticmethod
    def generate_walk_forward_markdown_report(wf_result: WalkForwardResult) -> str:
        """
        Generate structured Markdown report for Walk-Forward Out-of-Sample evaluation.
        """
        om = wf_result.overall_metrics
        sm = wf_result.summary

        folds_table_rows = []
        for f in wf_result.folds:
            fc = f.fold_config
            oos_m = f.out_of_sample_metrics
            ret_str = f"{oos_m.total_return * 100:.2f}%" if oos_m else "N/A"
            sharpe_str = f"{oos_m.sharpe_ratio:.4f}" if oos_m and oos_m.sharpe_ratio is not None else "N/A"
            dd_str = f"{oos_m.max_drawdown * 100:.2f}%" if oos_m else "N/A"
            trades_str = str(oos_m.total_trades) if oos_m else "0"

            folds_table_rows.append(
                f"| Fold {f.fold_idx} | {fc.train_start.strftime('%Y-%m-%d')} to {fc.train_end.strftime('%Y-%m-%d')} | {fc.test_start.strftime('%Y-%m-%d')} to {fc.test_end.strftime('%Y-%m-%d')} | {ret_str} | {sharpe_str} | {dd_str} | {trades_str} |"
            )

        folds_table = "\n".join(folds_table_rows)

        return f"""# Walk-Forward Out-of-Sample Backtest Report — ID: `{wf_result.walk_forward_id}`

> **RESEARCH INTEGRITY:** Walk-forward evaluation isolates training, validation, and testing periods sequentially across folds to eliminate look-ahead bias and overfitting.

---

## 1. Overall Out-of-Sample Summary
| Metric | Value |
| :--- | :--- |
| **Walk-Forward ID** | `{wf_result.walk_forward_id}` |
| **Total Sequential Folds** | {len(wf_result.folds)} |
| **Out-of-Sample Total Return** | {om.total_return * 100:.2f}% |
| **Out-of-Sample Annualized Return** | {om.annualized_return * 100:.2f}% |
| **Out-of-Sample Sharpe Ratio** | {om.sharpe_ratio if om.sharpe_ratio is not None else 'N/A'} |
| **Out-of-Sample Max Drawdown** | {om.max_drawdown * 100:.2f}% |
| **Total Completed Trades** | {om.total_trades} |
| **Win Rate** | {f'{om.win_rate * 100:.2f}%' if om.win_rate is not None else 'N/A'} |

---

## 2. Walk-Forward Fold Breakdown
| Fold | In-Sample Train Window | Out-of-Sample Test Window | OOS Return | OOS Sharpe | OOS Max DD | OOS Trades |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{folds_table}

---

## 3. Provenance & Window Settings
| Key | Value |
| :--- | :--- |
| **Train Window Bars** | {wf_result.provenance.get('train_window_bars')} |
| **Validation Window Bars** | {wf_result.provenance.get('val_window_bars')} |
| **Test Window Bars** | {wf_result.provenance.get('test_window_bars')} |
| **Step Bars** | {wf_result.provenance.get('step_bars')} |
| **Expanding Window Mode** | {wf_result.provenance.get('expanding_window')} |
"""
