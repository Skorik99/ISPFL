import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cs_sampling import build_selector_context, sample_clients


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _append_text(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def _write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_csv_matrix(
    path: str,
    matrix: np.ndarray,
    row_labels: Sequence[object],
    column_labels: Sequence[object],
    row_name: str,
) -> None:
    frame = pd.DataFrame(matrix, index=list(row_labels), columns=list(column_labels))
    frame.index.name = row_name
    frame.to_csv(path)


def _write_series_csv(
    path: str,
    values: Sequence[float],
    index_values: Sequence[object],
    index_name: str,
    value_name: str,
) -> None:
    frame = pd.DataFrame({index_name: list(index_values), value_name: list(values)})
    frame.to_csv(path, index=False)


def _normalize_backend(backend: str) -> str:
    backend = str(backend).lower()
    if backend not in {"thread", "process", "sequential"}:
        raise ValueError(
            "sampling_statistics.backend must be one of ['thread', 'process', 'sequential']"
        )
    return backend


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _format_subset_line(rep_id: int, subset: Sequence[int]) -> str:
    members = " ".join(str(int(client_id)) for client_id in subset)
    return f"{rep_id}\t{members}\n"


def _chunk_ranges(total_size: int, chunk_size: int) -> List[Tuple[int, int]]:
    if total_size <= 0:
        return []
    chunk_size = max(1, int(chunk_size))
    ranges = []
    for start_idx in range(0, total_size, chunk_size):
        end_idx = min(start_idx + chunk_size, total_size)
        ranges.append((start_idx, end_idx))
    return ranges


def _sample_subset_chunk_worker(
    selector_context: Dict[str, object],
    subset_size: int,
    amount_of_clients: int,
    client_pool: Optional[Sequence[int]],
    num_trials: int,
    rep_offset: int,
    seed: int,
    log_raw: bool,
    fedcor_tau: float,
    pow_topk_tau: float,
):
    rng = _make_rng(seed)
    counts = np.zeros(amount_of_clients, dtype=np.int64)
    subsets: List[List[int]] = []
    raw_lines: List[str] = []

    for local_rep in range(int(num_trials)):
        subset = sample_clients(
            selector_context,
            subset_size,
            client_pool=client_pool,
            rng=rng,
            fedcor_tau=fedcor_tau,
            pow_topk_tau=pow_topk_tau,
        )
        subset = sorted(int(client_id) for client_id in subset)
        counts[np.asarray(subset, dtype=np.int64)] += 1
        subsets.append(subset)
        if log_raw:
            raw_lines.append(_format_subset_line(rep_offset + local_rep, subset))

    return {
        "rep_offset": rep_offset,
        "counts": counts,
        "subsets": subsets,
        "raw_lines": raw_lines,
    }


def _estimate_psi_for_pool_worker(
    selector_context: Dict[str, object],
    amount_of_clients: int,
    m_grid: Sequence[int],
    p_matrix: np.ndarray,
    client_pool: Sequence[int],
    subset_index: int,
    b_value: int,
    t_in: int,
    seed: int,
    log_raw: bool,
    output_dir: str,
    fedcor_tau: float,
    pow_topk_tau: float,
):
    os.makedirs(output_dir, exist_ok=True)
    pool_array = np.asarray(
        sorted(int(client_id) for client_id in client_pool), dtype=np.int64
    )
    k_grid = [int(m_value) for m_value in m_grid if int(m_value) <= int(b_value)]

    psi_counts = np.zeros((len(k_grid), amount_of_clients), dtype=np.int64)
    raw_dir = os.path.join(output_dir, "raw_psi_samples")
    if log_raw:
        os.makedirs(raw_dir, exist_ok=True)

    pool_mask = np.zeros(amount_of_clients, dtype=np.float64)
    pool_mask[pool_array] = 1.0
    cap_values = p_matrix @ pool_mask
    d_values = np.full(len(k_grid), np.nan, dtype=np.float64)

    rng = _make_rng(seed)
    for k_idx, k_value in enumerate(k_grid):
        raw_lines: List[str] = []
        for trial_idx in range(int(t_in)):
            subset = sample_clients(
                selector_context,
                k_value,
                client_pool=pool_array,
                rng=rng,
                fedcor_tau=fedcor_tau,
                pow_topk_tau=pow_topk_tau,
            )
            subset = sorted(int(client_id) for client_id in subset)
            psi_counts[k_idx, np.asarray(subset, dtype=np.int64)] += 1
            if log_raw:
                raw_lines.append(_format_subset_line(trial_idx, subset))

        if log_raw:
            _write_text(
                os.path.join(raw_dir, f"k_{int(k_value):03d}.txt"),
                "".join(raw_lines),
            )

    psi_hat = psi_counts.astype(np.float64) / float(t_in)
    q_hat = np.zeros_like(psi_hat, dtype=np.float64)
    for row_idx, k_value in enumerate(k_grid):
        q_hat[row_idx] = psi_hat[row_idx] / float(k_value)
        m_index = list(m_grid).index(k_value)
        d_values[row_idx] = float(np.square(q_hat[row_idx] - p_matrix[m_index]).sum())

    client_columns = [f"client_{idx:03d}" for idx in range(amount_of_clients)]
    _write_csv_matrix(
        os.path.join(output_dir, "psi_counts.csv"),
        psi_counts,
        k_grid,
        client_columns,
        "k",
    )
    _write_csv_matrix(
        os.path.join(output_dir, "psi_hat.csv"),
        psi_hat,
        k_grid,
        client_columns,
        "k",
    )
    _write_csv_matrix(
        os.path.join(output_dir, "q_hat.csv"),
        q_hat,
        k_grid,
        client_columns,
        "k",
    )
    _write_series_csv(
        os.path.join(output_dir, "distance_by_k.csv"),
        d_values,
        k_grid,
        "k",
        "D",
    )
    _write_series_csv(
        os.path.join(output_dir, "capture_by_m.csv"),
        cap_values,
        m_grid,
        "m",
        "Cap",
    )
    _write_text(
        os.path.join(output_dir, "subset_members.txt"),
        " ".join(str(int(client_id)) for client_id in pool_array) + "\n",
    )
    np.save(os.path.join(output_dir, "subset_members.npy"), pool_array)
    np.save(os.path.join(output_dir, "psi_counts.npy"), psi_counts)
    np.save(os.path.join(output_dir, "psi_hat.npy"), psi_hat)
    np.save(os.path.join(output_dir, "q_hat.npy"), q_hat)

    summary_payload = {
        "subset_index": int(subset_index),
        "b": int(b_value),
        "k_grid": [int(value) for value in k_grid],
        "t_in": int(t_in),
        "seed": int(seed),
        "subset_members": pool_array.tolist(),
        "cap_values": cap_values.tolist(),
        "distance_values": d_values.tolist(),
    }
    _write_json(os.path.join(output_dir, "summary.json"), summary_payload)
    _write_text(
        os.path.join(output_dir, "summary.txt"),
        json.dumps(summary_payload, indent=2, sort_keys=True),
    )

    return {
        "subset_index": int(subset_index),
        "k_grid": k_grid,
        "cap_values": cap_values,
        "d_values": d_values,
    }


class ClientSelectionStatisticsCollector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stats_cfg = cfg.sampling_statistics
        self.enabled = bool(getattr(self.stats_cfg, "enabled", False))
        self.backend = _normalize_backend(self.stats_cfg.backend)
        self.output_root = os.path.join(
            cfg.single_run_dir, str(self.stats_cfg.output_subdir)
        )
        self.master_log_path = os.path.join(self.output_root, "master_log.txt")
        self.collection_counter = 0
        self.client_columns: Optional[List[str]] = None

        if self.enabled:
            os.makedirs(self.output_root, exist_ok=True)
            self._init_master_log()

    def _init_master_log(self) -> None:
        payload = {
            "backend": self.backend,
            "num_workers": int(self.stats_cfg.num_workers),
            "chunk_size": int(self.stats_cfg.chunk_size),
            "R_pi": int(self.stats_cfg.R_pi),
            "R_b": int(self.stats_cfg.R_b),
            "T_in": int(self.stats_cfg.T_in),
            "seed_offset": int(self.stats_cfg.seed_offset),
        }
        _write_text(
            self.master_log_path,
            "Client selection statistics collector initialized\n"
            + json.dumps(payload, indent=2, sort_keys=True)
            + "\n\n",
        )

    def collect(self, method) -> None:
        if not self.enabled:
            return
        collection_started = time.time()
        cur_round = int(getattr(method, "cur_round", 0))
        amount_of_clients = int(getattr(method, "amount_of_clients"))
        self.client_columns = [f"client_{idx:03d}" for idx in range(amount_of_clients)]
        selector_context = build_selector_context(method)
        selector_name = str(selector_context["selector_name"]).lower()

        event_dir = os.path.join(
            self.output_root,
            f"collection_{self.collection_counter:04d}_round_{cur_round:04d}_{selector_name}",
        )
        self.collection_counter += 1
        os.makedirs(event_dir, exist_ok=True)

        m_grid = self._resolve_grid(
            amount_of_clients=amount_of_clients,
            values=self.stats_cfg.m_grid_values,
            start=int(self.stats_cfg.m_grid_start),
            stop=self.stats_cfg.m_grid_stop,
            step=int(self.stats_cfg.m_grid_step),
            include_stop=bool(self.stats_cfg.m_grid_include_stop),
        )
        b_grid = self._resolve_b_grid(amount_of_clients, m_grid)
        if not m_grid:
            raise ValueError("sampling_statistics.m_grid resolved to an empty list")
        if not b_grid:
            raise ValueError("sampling_statistics.b_grid resolved to an empty list")

        self._persist_selector_snapshot(event_dir, selector_context)
        self._write_collection_metadata(
            event_dir=event_dir,
            method=method,
            selector_name=selector_name,
            m_grid=m_grid,
            b_grid=b_grid,
        )

        pi_counts = np.zeros((len(m_grid), amount_of_clients), dtype=np.int64)
        pi_hat = np.zeros((len(m_grid), amount_of_clients), dtype=np.float64)

        for m_idx, m_value in enumerate(m_grid):
            counts, _ = self._estimate_subset_statistics(
                selector_context=selector_context,
                subset_size=int(m_value),
                client_pool=None,
                num_trials=int(self.stats_cfg.R_pi),
                stage_seed=self._event_seed(cur_round, 1000 + m_idx),
                raw_output_path=(
                    os.path.join(event_dir, f"pi_raw_samples_m_{int(m_value):03d}.txt")
                    if self.stats_cfg.log_pi_samples
                    else None
                ),
            )
            pi_counts[m_idx] = counts
            pi_hat[m_idx] = counts.astype(np.float64) / float(self.stats_cfg.R_pi)

        p_matrix = np.zeros_like(pi_hat, dtype=np.float64)
        for m_idx, m_value in enumerate(m_grid):
            p_matrix[m_idx] = pi_hat[m_idx] / float(m_value)
        ess_values = 1.0 / np.maximum(np.square(p_matrix).sum(axis=1), 1e-18)

        self._persist_pi_outputs(
            event_dir, m_grid, pi_counts, pi_hat, p_matrix, ess_values
        )

        cap_mean_rows = []
        d_mean_rows = []

        for b_idx, b_value in enumerate(b_grid):
            b_dir = os.path.join(event_dir, f"b_{int(b_value):03d}")
            os.makedirs(b_dir, exist_ok=True)

            _, i_subsets = self._estimate_subset_statistics(
                selector_context=selector_context,
                subset_size=int(b_value),
                client_pool=None,
                num_trials=int(self.stats_cfg.R_b),
                stage_seed=self._event_seed(cur_round, 2000 + b_idx),
                raw_output_path=(
                    os.path.join(b_dir, "I_subsets.txt")
                    if self.stats_cfg.log_i_samples
                    else None
                ),
                keep_subsets=True,
            )

            if i_subsets is None:
                raise RuntimeError("Expected sampled I subsets to be returned")

            self._persist_i_subsets(
                b_dir=b_dir,
                i_subsets=i_subsets,
                amount_of_clients=amount_of_clients,
            )

            cap_matrix = np.zeros((len(i_subsets), len(m_grid)), dtype=np.float64)
            for subset_idx, subset in enumerate(i_subsets):
                cap_matrix[subset_idx] = p_matrix @ self._subset_mask(
                    subset, amount_of_clients
                )

            k_grid = [
                int(m_value) for m_value in m_grid if int(m_value) <= int(b_value)
            ]
            d_matrix = np.full((len(i_subsets), len(k_grid)), np.nan, dtype=np.float64)

            psi_results = self._estimate_psi_for_b(
                selector_context=selector_context,
                amount_of_clients=amount_of_clients,
                m_grid=m_grid,
                p_matrix=p_matrix,
                i_subsets=i_subsets,
                b_value=int(b_value),
                b_dir=b_dir,
                cur_round=cur_round,
            )

            for result in psi_results:
                subset_index = int(result["subset_index"])
                d_matrix[subset_index, :] = np.asarray(
                    result["d_values"], dtype=np.float64
                )

            cap_mean = cap_matrix.mean(axis=0)
            d_mean = np.nanmean(d_matrix, axis=0) if d_matrix.size else np.array([])

            cap_frame = pd.DataFrame(cap_matrix, columns=[f"m_{m:03d}" for m in m_grid])
            cap_frame.insert(0, "subset_index", np.arange(len(i_subsets)))
            cap_frame.to_csv(os.path.join(b_dir, "capture_by_subset.csv"), index=False)
            if len(k_grid) > 0:
                d_frame = pd.DataFrame(d_matrix, columns=[f"k_{k:03d}" for k in k_grid])
                d_frame.insert(0, "subset_index", np.arange(len(i_subsets)))
                d_frame.to_csv(
                    os.path.join(b_dir, "distance_by_subset.csv"), index=False
                )
            _write_series_csv(
                os.path.join(b_dir, "capture_mean_by_m.csv"),
                cap_mean,
                m_grid,
                "m",
                "Cap",
            )
            if len(k_grid) > 0:
                _write_series_csv(
                    os.path.join(b_dir, "distance_mean_by_k.csv"),
                    d_mean,
                    k_grid,
                    "k",
                    "D",
                )

            cap_mean_rows.extend(
                {"b": int(b_value), "m": int(m_value), "Cap": float(cap_mean[m_idx])}
                for m_idx, m_value in enumerate(m_grid)
            )
            d_mean_rows.extend(
                {"b": int(b_value), "m": int(k_value), "D": float(d_mean[k_idx])}
                for k_idx, k_value in enumerate(k_grid)
            )

        pd.DataFrame(cap_mean_rows).to_csv(
            os.path.join(event_dir, "capture_mean_all.csv"), index=False
        )
        pd.DataFrame(d_mean_rows).to_csv(
            os.path.join(event_dir, "distance_mean_all.csv"), index=False
        )

        elapsed = time.time() - collection_started
        summary_payload = {
            "round": cur_round,
            "selector_name": selector_name,
            "amount_of_clients": amount_of_clients,
            "m_grid": [int(value) for value in m_grid],
            "b_grid": [int(value) for value in b_grid],
            "elapsed_seconds": elapsed,
            "event_dir": event_dir,
        }
        _write_json(os.path.join(event_dir, "collection_summary.json"), summary_payload)
        _write_text(
            os.path.join(event_dir, "collection_summary.txt"),
            json.dumps(summary_payload, indent=2, sort_keys=True),
        )
        _append_text(
            self.master_log_path,
            f"[round={cur_round}] selector={selector_name} event_dir={event_dir} "
            f"elapsed={elapsed:.2f}s\n",
        )
        print(
            f"[SamplingStats] Round {cur_round}: collected selector statistics for {selector_name} "
            f"in {elapsed:.2f}s at {event_dir}"
        )

    def _estimate_psi_for_b(
        self,
        *,
        selector_context: Dict[str, object],
        amount_of_clients: int,
        m_grid: Sequence[int],
        p_matrix: np.ndarray,
        i_subsets: Sequence[Sequence[int]],
        b_value: int,
        b_dir: str,
        cur_round: int,
    ) -> List[Dict[str, object]]:
        tasks = []
        for subset_index, client_pool in enumerate(i_subsets):
            subset_dir = os.path.join(b_dir, f"I_{subset_index:04d}")
            seed = self._event_seed(cur_round, 3000 + b_value * 1000 + subset_index)
            tasks.append(
                (
                    subset_index,
                    client_pool,
                    subset_dir,
                    seed,
                )
            )

        if self.backend == "sequential" or len(tasks) <= 1:
            return [
                _estimate_psi_for_pool_worker(
                    selector_context=selector_context,
                    amount_of_clients=amount_of_clients,
                    m_grid=m_grid,
                    p_matrix=p_matrix,
                    client_pool=client_pool,
                    subset_index=subset_index,
                    b_value=b_value,
                    t_in=int(self.stats_cfg.T_in),
                    seed=seed,
                    log_raw=bool(self.stats_cfg.log_psi_samples),
                    output_dir=subset_dir,
                    fedcor_tau=float(self.stats_cfg.fedcor_tau),
                    pow_topk_tau=float(self.stats_cfg.pow_topk_tau),
                )
                for subset_index, client_pool, subset_dir, seed in tasks
            ]

        executor_cls = (
            ProcessPoolExecutor if self.backend == "process" else ThreadPoolExecutor
        )
        results: List[Dict[str, object]] = []
        with executor_cls(max_workers=int(self.stats_cfg.num_workers)) as executor:
            future_map = {
                executor.submit(
                    _estimate_psi_for_pool_worker,
                    selector_context,
                    amount_of_clients,
                    m_grid,
                    p_matrix,
                    client_pool,
                    subset_index,
                    b_value,
                    int(self.stats_cfg.T_in),
                    seed,
                    bool(self.stats_cfg.log_psi_samples),
                    subset_dir,
                    float(self.stats_cfg.fedcor_tau),
                    float(self.stats_cfg.pow_topk_tau),
                ): subset_index
                for subset_index, client_pool, subset_dir, seed in tasks
            }
            for future in as_completed(future_map):
                results.append(future.result())

        results.sort(key=lambda item: int(item["subset_index"]))
        return results

    def _persist_selector_snapshot(
        self, event_dir: str, selector_context: Dict[str, object]
    ) -> None:
        snapshot_dir = os.path.join(event_dir, "selector_snapshot")
        os.makedirs(snapshot_dir, exist_ok=True)
        meta_payload = {}
        for key, value in selector_context.items():
            if isinstance(value, np.ndarray):
                npy_path = os.path.join(snapshot_dir, f"{key}.npy")
                np.save(npy_path, value)
                csv_path = os.path.join(snapshot_dir, f"{key}.csv")
                if value.ndim == 1:
                    pd.DataFrame({key: value}).to_csv(csv_path, index=False)
                elif value.ndim == 2:
                    pd.DataFrame(value).to_csv(csv_path, index=False)
                else:
                    np.savetxt(
                        os.path.join(snapshot_dir, f"{key}.txt"),
                        value.reshape(-1, 1),
                        fmt="%.10g",
                    )
                meta_payload[key] = {
                    "type": "ndarray",
                    "shape": list(value.shape),
                    "path": f"selector_snapshot/{key}.npy",
                }
            else:
                meta_payload[key] = value
        _write_json(os.path.join(snapshot_dir, "metadata.json"), meta_payload)
        _write_text(
            os.path.join(snapshot_dir, "metadata.txt"),
            json.dumps(meta_payload, indent=2, sort_keys=True),
        )

    def _write_collection_metadata(
        self,
        *,
        event_dir: str,
        method,
        selector_name: str,
        m_grid: Sequence[int],
        b_grid: Sequence[int],
    ) -> None:
        payload = {
            "selector_name": selector_name,
            "round": int(getattr(method, "cur_round", 0)),
            "amount_of_clients": int(getattr(method, "amount_of_clients", 0)),
            "m_grid": [int(value) for value in m_grid],
            "b_grid": [int(value) for value in b_grid],
            "R_pi": int(self.stats_cfg.R_pi),
            "R_b": int(self.stats_cfg.R_b),
            "T_in": int(self.stats_cfg.T_in),
            "backend": self.backend,
            "num_workers": int(self.stats_cfg.num_workers),
            "chunk_size": int(self.stats_cfg.chunk_size),
            "fedcor_tau": float(self.stats_cfg.fedcor_tau),
            "pow_topk_tau": float(self.stats_cfg.pow_topk_tau),
        }
        _write_json(os.path.join(event_dir, "metadata.json"), payload)
        _write_text(
            os.path.join(event_dir, "metadata.txt"),
            json.dumps(payload, indent=2, sort_keys=True),
        )

    def _persist_pi_outputs(
        self,
        event_dir: str,
        m_grid: Sequence[int],
        pi_counts: np.ndarray,
        pi_hat: np.ndarray,
        p_matrix: np.ndarray,
        ess_values: np.ndarray,
    ) -> None:
        if self.client_columns is None:
            raise RuntimeError(
                "client_columns must be initialized before saving outputs"
            )

        _write_csv_matrix(
            os.path.join(event_dir, "pi_counts.csv"),
            pi_counts,
            m_grid,
            self.client_columns,
            "m",
        )
        _write_csv_matrix(
            os.path.join(event_dir, "pi_hat.csv"),
            pi_hat,
            m_grid,
            self.client_columns,
            "m",
        )
        _write_csv_matrix(
            os.path.join(event_dir, "p_hat.csv"),
            p_matrix,
            m_grid,
            self.client_columns,
            "m",
        )
        _write_series_csv(
            os.path.join(event_dir, "ess.csv"),
            ess_values,
            m_grid,
            "m",
            "ESS",
        )

        np.save(os.path.join(event_dir, "pi_counts.npy"), pi_counts)
        np.save(os.path.join(event_dir, "pi_hat.npy"), pi_hat)
        np.save(os.path.join(event_dir, "p_hat.npy"), p_matrix)
        np.save(os.path.join(event_dir, "ess.npy"), ess_values)

    def _persist_i_subsets(
        self,
        *,
        b_dir: str,
        i_subsets: Sequence[Sequence[int]],
        amount_of_clients: int,
    ) -> None:
        subset_members_dir = os.path.join(b_dir, "I_subsets")
        os.makedirs(subset_members_dir, exist_ok=True)

        inclusion_counts = np.zeros(amount_of_clients, dtype=np.int64)
        raw_lines = []
        for subset_index, subset in enumerate(i_subsets):
            subset_array = np.asarray(
                sorted(int(client_id) for client_id in subset), dtype=np.int64
            )
            inclusion_counts[subset_array] += 1
            raw_lines.append(_format_subset_line(subset_index, subset_array))
            _write_text(
                os.path.join(subset_members_dir, f"I_{subset_index:04d}.txt"),
                " ".join(str(int(client_id)) for client_id in subset_array) + "\n",
            )
            np.save(
                os.path.join(subset_members_dir, f"I_{subset_index:04d}.npy"),
                subset_array,
            )

        _write_text(os.path.join(b_dir, "I_subsets_verbose.txt"), "".join(raw_lines))
        _write_series_csv(
            os.path.join(b_dir, "I_inclusion_counts.csv"),
            inclusion_counts,
            list(range(amount_of_clients)),
            "client",
            "count",
        )
        _write_series_csv(
            os.path.join(b_dir, "I_inclusion_hat.csv"),
            inclusion_counts.astype(np.float64) / max(1, len(i_subsets)),
            list(range(amount_of_clients)),
            "client",
            "probability",
        )

    def _estimate_subset_statistics(
        self,
        *,
        selector_context: Dict[str, object],
        subset_size: int,
        client_pool: Optional[Sequence[int]],
        num_trials: int,
        stage_seed: int,
        raw_output_path: Optional[str],
        keep_subsets: bool = False,
    ) -> Tuple[np.ndarray, Optional[List[List[int]]]]:
        amount_of_clients = int(selector_context["amount_of_clients"])
        chunk_ranges = _chunk_ranges(num_trials, int(self.stats_cfg.chunk_size))
        if not chunk_ranges:
            return np.zeros(amount_of_clients, dtype=np.int64), (
                [] if keep_subsets else None
            )

        if self.backend == "sequential" or len(chunk_ranges) == 1:
            results = [
                _sample_subset_chunk_worker(
                    selector_context=selector_context,
                    subset_size=subset_size,
                    amount_of_clients=amount_of_clients,
                    client_pool=client_pool,
                    num_trials=end_idx - start_idx,
                    rep_offset=start_idx,
                    seed=stage_seed + start_idx,
                    log_raw=raw_output_path is not None,
                    fedcor_tau=float(self.stats_cfg.fedcor_tau),
                    pow_topk_tau=float(self.stats_cfg.pow_topk_tau),
                )
                for start_idx, end_idx in chunk_ranges
            ]
        else:
            executor_cls = (
                ProcessPoolExecutor if self.backend == "process" else ThreadPoolExecutor
            )
            results = []
            with executor_cls(max_workers=int(self.stats_cfg.num_workers)) as executor:
                future_map = {
                    executor.submit(
                        _sample_subset_chunk_worker,
                        selector_context,
                        subset_size,
                        amount_of_clients,
                        client_pool,
                        end_idx - start_idx,
                        start_idx,
                        stage_seed + start_idx,
                        raw_output_path is not None,
                        float(self.stats_cfg.fedcor_tau),
                        float(self.stats_cfg.pow_topk_tau),
                    ): start_idx
                    for start_idx, end_idx in chunk_ranges
                }
                for future in as_completed(future_map):
                    results.append(future.result())

        results.sort(key=lambda item: int(item["rep_offset"]))
        counts = np.zeros(amount_of_clients, dtype=np.int64)
        subsets: Optional[List[List[int]]] = [] if keep_subsets else None
        raw_handle = None
        if raw_output_path is not None:
            raw_handle = open(raw_output_path, "w", encoding="utf-8")
        try:
            for result in results:
                counts += np.asarray(result["counts"], dtype=np.int64)
                if keep_subsets and subsets is not None:
                    subsets.extend(result["subsets"])
                if raw_handle is not None:
                    raw_handle.writelines(result["raw_lines"])
        finally:
            if raw_handle is not None:
                raw_handle.close()

        return counts, subsets

    def _subset_mask(self, subset: Sequence[int], amount_of_clients: int) -> np.ndarray:
        mask = np.zeros(amount_of_clients, dtype=np.float64)
        mask[np.asarray(subset, dtype=np.int64)] = 1.0
        return mask

    def _resolve_b_grid(
        self, amount_of_clients: int, m_grid: Sequence[int]
    ) -> List[int]:
        if self.stats_cfg.b_grid_values is not None:
            values = [int(value) for value in self.stats_cfg.b_grid_values]
        elif bool(self.stats_cfg.b_grid_use_m_grid):
            values = [int(value) for value in m_grid]
        else:
            values = self._resolve_grid(
                amount_of_clients=amount_of_clients,
                values=None,
                start=int(self.stats_cfg.b_grid_start),
                stop=self.stats_cfg.b_grid_stop,
                step=int(self.stats_cfg.b_grid_step),
                include_stop=bool(self.stats_cfg.b_grid_include_stop),
            )

        values = [value for value in values if 1 <= value <= amount_of_clients]
        return sorted(set(values))

    def _resolve_grid(
        self,
        *,
        amount_of_clients: int,
        values,
        start: int,
        stop,
        step: int,
        include_stop: bool,
    ) -> List[int]:
        if values is not None:
            grid = [int(value) for value in values]
        else:
            stop_value = amount_of_clients if stop is None else int(stop)
            if include_stop:
                stop_value += 1
            grid = list(range(int(start), stop_value, int(step)))
        grid = [value for value in grid if 1 <= value <= amount_of_clients]
        return sorted(set(grid))

    def _event_seed(self, cur_round: int, salt: int) -> int:
        return (
            int(self.cfg.random_state)
            + int(self.stats_cfg.seed_offset)
            + int(cur_round) * 100_000
            + int(salt)
        )
