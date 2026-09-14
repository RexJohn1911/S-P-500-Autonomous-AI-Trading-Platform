"""
Market Regime Analytics, Transition Matrix, Duration Analysis, and Clustering Diagnostics.
Computes empirical Markov transition dynamics, regime persistence, run-length durations, and cluster validity metrics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from backend.app.regime.schemas import ClusteringDiagnostics, RegimeStatistics, TransitionMatrix


class RegimeAnalytics:
    """
    Quantitative analysis toolkit for market regimes.
    Calculates transition matrices, run-length durations, return/volatility distributions, and clustering quality.
    """

    @staticmethod
    def compute_transition_matrix(
        regime_sequence: Union[List[int], np.ndarray],
        semantic_mapping: Dict[int, str],
    ) -> TransitionMatrix:
        """
        Compute empirical transition matrix P(S_{t+1} | S_t) from a chronological sequence of regime IDs.
        """
        seq = np.asarray(regime_sequence, dtype=int)
        if len(seq) < 2:
            regime_names = [semantic_mapping.get(k, f"REGIME_{k}") for k in sorted(semantic_mapping.keys())]
            n_states = len(regime_names)
            identity = np.eye(n_states).tolist()
            zeros = np.zeros((n_states, n_states), dtype=int).tolist()
            return TransitionMatrix(
                regime_names=regime_names,
                matrix=identity,
                transition_counts=zeros,
                total_transitions=0,
                persistence_probabilities={name: 1.0 for name in regime_names},
            )

        unique_ids = sorted(list(semantic_mapping.keys()))
        id_to_idx = {rid: i for i, rid in enumerate(unique_ids)}
        regime_names = [semantic_mapping.get(rid, f"REGIME_{rid}") for rid in unique_ids]
        n_states = len(unique_ids)

        counts = np.zeros((n_states, n_states), dtype=int)
        for t in range(len(seq) - 1):
            curr_id = seq[t]
            next_id = seq[t + 1]
            if curr_id in id_to_idx and next_id in id_to_idx:
                counts[id_to_idx[curr_id], id_to_idx[next_id]] += 1

        # Normalize rows to probabilities
        prob_matrix = np.zeros((n_states, n_states), dtype=float)
        persistence = {}
        for i in range(n_states):
            row_sum = np.sum(counts[i])
            if row_sum > 0:
                prob_matrix[i] = counts[i] / row_sum
            else:
                # If a state was never transitioned from, set self-persistence to 1.0
                prob_matrix[i, i] = 1.0
            persistence[regime_names[i]] = float(prob_matrix[i, i])

        return TransitionMatrix(
            regime_names=regime_names,
            matrix=prob_matrix.tolist(),
            transition_counts=counts.tolist(),
            total_transitions=int(np.sum(counts)),
            persistence_probabilities=persistence,
        )

    @staticmethod
    def calculate_regime_durations(regime_sequence: Union[List[int], np.ndarray]) -> Dict[int, List[int]]:
        """
        Extract run-length consecutive bar durations for each regime in a chronological sequence.
        """
        seq = np.asarray(regime_sequence, dtype=int)
        if len(seq) == 0:
            return {}

        durations: Dict[int, List[int]] = {}
        current_regime = seq[0]
        current_len = 1

        for i in range(1, len(seq)):
            if seq[i] == current_regime:
                current_len += 1
            else:
                durations.setdefault(current_regime, []).append(current_len)
                current_regime = seq[i]
                current_len = 1
        durations.setdefault(current_regime, []).append(current_len)
        return durations

    @classmethod
    def compute_regime_statistics(
        cls,
        regime_sequence: Union[List[int], np.ndarray],
        returns: Union[List[float], np.ndarray],
        volatilities: Optional[Union[List[float], np.ndarray]] = None,
        vix_series: Optional[Union[List[float], np.ndarray]] = None,
        semantic_mapping: Optional[Dict[int, str]] = None,
    ) -> Dict[str, RegimeStatistics]:
        """
        Calculate descriptive empirical statistics across all detected regimes.
        """
        seq = np.asarray(regime_sequence, dtype=int)
        rets = np.asarray(returns, dtype=float)
        n = len(seq)
        if n == 0 or len(rets) != n:
            return {}

        vols = np.asarray(volatilities, dtype=float) if volatilities is not None else np.abs(rets)
        vix = np.asarray(vix_series, dtype=float) if vix_series is not None else None
        mapping = semantic_mapping or {}

        durations_by_regime = cls.calculate_regime_durations(seq)
        unique_ids = sorted(list(set(seq).union(mapping.keys())))

        stats_dict = {}
        for rid in unique_ids:
            mask = (seq == rid)
            count = int(np.sum(mask))
            reg_name = mapping.get(rid, f"REGIME_{rid}")

            if count > 0:
                pct = count / n
                reg_rets = rets[mask]
                valid_rets = reg_rets[~np.isnan(reg_rets)]
                mean_ret = float(np.mean(valid_rets)) if len(valid_rets) > 0 else 0.0
                median_ret = float(np.median(valid_rets)) if len(valid_rets) > 0 else 0.0
                std_ret = float(np.std(valid_rets)) if len(valid_rets) > 0 else 0.0

                reg_vols = vols[mask]
                valid_vols = reg_vols[~np.isnan(reg_vols)]
                mean_vol = float(np.mean(valid_vols)) if len(valid_vols) > 0 else 0.0
                median_vol = float(np.median(valid_vols)) if len(valid_vols) > 0 else 0.0

                mean_vix_val = None
                if vix is not None:
                    vix_sub = vix[mask]
                    valid_vix = vix_sub[~np.isnan(vix_sub)]
                    mean_vix_val = float(np.mean(valid_vix)) if len(valid_vix) > 0 else None

                reg_durations = durations_by_regime.get(rid, [1])
                mean_dur = float(np.mean(reg_durations)) if len(reg_durations) > 0 else 0.0
                median_dur = float(np.median(reg_durations)) if len(reg_durations) > 0 else 0.0
                min_dur = int(np.min(reg_durations)) if len(reg_durations) > 0 else 0
                max_dur = int(np.max(reg_durations)) if len(reg_durations) > 0 else 0
            else:
                pct = 0.0
                mean_ret = median_ret = std_ret = mean_vol = median_vol = 0.0
                mean_vix_val = None
                mean_dur = median_dur = 0.0
                min_dur = max_dur = 0

            stats_dict[reg_name] = RegimeStatistics(
                regime_id=rid,
                regime_name=reg_name,
                observation_count=count,
                percentage_of_samples=pct,
                mean_return=mean_ret,
                median_return=median_ret,
                return_std=std_ret,
                mean_volatility=mean_vol,
                median_volatility=median_vol,
                mean_vix=mean_vix_val,
                mean_duration_bars=mean_dur,
                median_duration_bars=median_dur,
                min_duration_bars=min_dur,
                max_duration_bars=max_dur,
            )

        return stats_dict

    @staticmethod
    def compute_clustering_diagnostics(
        X: np.ndarray,
        labels: np.ndarray,
        bic: Optional[float] = None,
        aic: Optional[float] = None,
        log_likelihood: Optional[float] = None,
    ) -> ClusteringDiagnostics:
        """
        Compute standard unsupervised clustering validity scores (Silhouette, Davies-Bouldin, Calinski-Harabasz).
        """
        n_samples = len(X)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Silhouette & cluster validity requires at least 2 clusters and more samples than clusters
        if n_samples < 4 or n_clusters < 2 or n_clusters >= n_samples:
            return ClusteringDiagnostics(
                bic=bic,
                aic=aic,
                log_likelihood=log_likelihood,
            )

        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = None

        try:
            db = float(davies_bouldin_score(X, labels))
        except Exception:
            db = None

        try:
            ch = float(calinski_harabasz_score(X, labels))
        except Exception:
            ch = None

        return ClusteringDiagnostics(
            silhouette_score=sil,
            davies_bouldin_index=db,
            calinski_harabasz_score=ch,
            bic=bic,
            aic=aic,
            log_likelihood=log_likelihood,
        )
