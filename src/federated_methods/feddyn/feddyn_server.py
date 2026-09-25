import csv
import math
from pathlib import Path
from collections import OrderedDict

import torch

from ..fedavg.fedavg_server import FedAvgServer


class FedDynServer(FedAvgServer):
    SERVER_DEBUG_FIELDS = [
        "round",
        "context",
        "selected_clients",
        "dual_state_scale",
        "correction_max_ratio",
        "parameter_update_norm",
        "dual_correction_norm_raw",
        "dual_correction_norm_applied",
        "correction_scale",
        "update_correction_cosine",
        "running_mean_update_norm",
        "running_var_update_norm",
        "num_batches_tracked_update_norm",
        "other_buffer_update_norm",
    ]

    def __init__(self, cfg, alpha, server_correction_max_ratio=0.85):
        super().__init__(cfg)
        self.alpha = float(alpha)
        self.server_correction_max_ratio = float(server_correction_max_ratio)
        self.client_dual_states = None
        self.parameter_names = []
        self.buffer_names = set()
        self.dual_state_scale = 1.0
        self.debug_dual_state = bool(cfg.federated_method.debug_dual_state)
        self.dual_state_debug_file = None
        self.dual_state_debug_writer = None
        self.server_debug_file = None
        self.server_debug_writer = None
        self._isp_correction_cache = None

        if self.debug_dual_state:
            debug_path = Path(cfg.single_run_dir) / "feddyn_dual_state.csv"
            self.dual_state_debug_file = debug_path.open("w", newline="")
            self.dual_state_debug_writer = csv.DictWriter(
                self.dual_state_debug_file,
                fieldnames=[
                    "round",
                    "client_id",
                    "client_size",
                    "alpha_adaptive",
                    "dual_state_scale",
                    "dual_state_norm",
                    "dual_correction_norm",
                    "local_update_norm",
                    "r_i",
                ],
            )
            self.dual_state_debug_writer.writeheader()
            server_debug_path = (
                Path(cfg.single_run_dir) / "feddyn_server_correction.csv"
            )
            self.server_debug_file = server_debug_path.open("w", newline="")
            self.server_debug_writer = csv.DictWriter(
                self.server_debug_file,
                fieldnames=self.SERVER_DEBUG_FIELDS,
            )
            self.server_debug_writer.writeheader()

    def initialize_dual_states(self, global_model):
        self.parameter_names = [
            name for name, param in global_model.named_parameters() if param.requires_grad
        ]
        self.buffer_names = {name for name, _ in global_model.named_buffers()}

        zero_state = OrderedDict(
            (
                name,
                torch.zeros_like(param.detach(), device="cpu"),
            )
            for name, param in global_model.named_parameters()
            if param.requires_grad
        )
        self.client_dual_states = [
            OrderedDict((name, tensor.clone()) for name, tensor in zero_state.items())
            for _ in range(self.amount_of_clients)
        ]

    def get_client_dual_state(self, rank):
        if self.client_dual_states is None:
            return None
        return {
            name: tensor.clone()
            for name, tensor in self.client_dual_states[rank].items()
        }

    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        if self.dual_state_debug_writer is not None:
            debug_row = client_result.get("dual_state_debug")
            if debug_row is not None:
                self.dual_state_debug_writer.writerow(
                    {"round": self.cur_round, **debug_row}
                )
        local_dual_state = client_result.get("local_dual_state")
        if local_dual_state is not None and self.client_dual_states is not None:
            self.client_dual_states[client_result["rank"]] = OrderedDict(
                (name, tensor.detach().cpu())
                for name, tensor in local_dual_state.items()
            )
            self._isp_correction_cache = None

    @staticmethod
    def _mean_cpu_tensors(tensors):
        total = tensors[0].detach().cpu().clone()
        if not total.is_floating_point():
            total = total.float()
        for tensor in tensors[1:]:
            total.add_(tensor.detach().cpu())
        total.mul_(1.0 / len(tensors))
        return total

    def _mean_client_updates(self, clients):
        first_update = self.client_gradients[clients[0]]
        return {
            key: self._mean_cpu_tensors(
                [self.client_gradients[client][key] for client in clients]
            ).to(self.device, non_blocking=True)
            for key in first_update
        }

    def _scaled_dual_mean(self, scale, use_cache=False):
        if use_cache and self._isp_correction_cache is not None:
            cached_scale, correction = self._isp_correction_cache
            if cached_scale == scale:
                return correction

        correction = {
            name: self._mean_cpu_tensors(
                [state[name] for state in self.client_dual_states]
            ).mul_(scale).to(self.device, non_blocking=True)
            for name in self.parameter_names
        }
        if use_cache:
            self._isp_correction_cache = (scale, correction)
        return correction

    def clear_isp_correction_cache(self):
        self._isp_correction_cache = None

    @staticmethod
    def _norm_from_squared(squared_norm):
        return torch.sqrt(squared_norm).item()

    def _correction_metrics(self, mean_updates, correction):
        device = next(iter(correction.values())).device
        update_squared = torch.zeros((), device=device)
        correction_squared = torch.zeros((), device=device)
        dot_product = (
            torch.zeros((), device=device)
            if self.server_debug_writer is not None
            else None
        )
        for name in self.parameter_names:
            update = mean_updates[name].reshape(-1)
            dual = correction[name].reshape(-1)
            update_squared.add_(torch.dot(update, update))
            correction_squared.add_(torch.dot(dual, dual))
            if dot_product is not None:
                dot_product.add_(torch.dot(update, dual))

        update_norm = self._norm_from_squared(update_squared)
        correction_norm = self._norm_from_squared(correction_squared)
        denominator = update_norm * correction_norm
        cosine = (
            dot_product.item() / denominator
            if dot_product is not None and denominator
            else 0.0
        )
        cosine = max(-1.0, min(1.0, cosine))
        correction_scale = min(
            1.0,
            self.server_correction_max_ratio
            * update_norm
            / max(correction_norm, 1e-12),
        )
        return update_norm, correction_norm, correction_scale, cosine

    def _buffer_update_norms(self, mean_updates):
        device = next(iter(mean_updates.values())).device
        categories = {
            "running_mean": torch.zeros((), device=device),
            "running_var": torch.zeros((), device=device),
            "num_batches_tracked": torch.zeros((), device=device),
            "other": torch.zeros((), device=device),
        }
        for name in self.buffer_names:
            if name.endswith("running_mean"):
                category = "running_mean"
            elif name.endswith("running_var"):
                category = "running_var"
            elif name.endswith("num_batches_tracked"):
                category = "num_batches_tracked"
            else:
                category = "other"
            update = mean_updates[name].float().reshape(-1)
            categories[category].add_(torch.dot(update, update))
        return {
            category: math.sqrt(squared_norm.item())
            for category, squared_norm in categories.items()
        }

    def _write_server_debug(
        self,
        context,
        clients,
        update_norm,
        correction_norm,
        correction_scale,
        cosine,
        buffer_norms,
        dual_scale,
    ):
        if self.server_debug_writer is None:
            return
        self.server_debug_writer.writerow(
            {
                "round": self.cur_round,
                "context": context,
                "selected_clients": len(clients),
                "dual_state_scale": dual_scale,
                "correction_max_ratio": self.server_correction_max_ratio,
                "parameter_update_norm": update_norm,
                "dual_correction_norm_raw": correction_norm,
                "dual_correction_norm_applied": correction_norm * correction_scale,
                "correction_scale": correction_scale,
                "update_correction_cosine": cosine,
                "running_mean_update_norm": buffer_norms["running_mean"],
                "running_var_update_norm": buffer_norms["running_var"],
                "num_batches_tracked_update_norm": buffer_norms[
                    "num_batches_tracked"
                ],
                "other_buffer_update_norm": buffer_norms["other"],
            }
        )
        if context == "train":
            self.server_debug_file.flush()

    def aggregate_from_state(
        self,
        base_state,
        clients,
        dual_scale,
        context,
        cache_correction=False,
    ):
        mean_updates = self._mean_client_updates(clients)
        aggregated_state = {
            key: value.detach().to(self.device).clone()
            for key, value in base_state.items()
        }
        for key, update in mean_updates.items():
            if aggregated_state[key].is_floating_point():
                aggregated_state[key].add_(update)
            else:
                aggregated_state[key] = aggregated_state[key] + update

        correction = self._scaled_dual_mean(
            dual_scale,
            use_cache=cache_correction,
        )
        update_norm, correction_norm, correction_scale, cosine = (
            self._correction_metrics(mean_updates, correction)
        )
        for name, dual in correction.items():
            aggregated_state[name].add_(dual, alpha=correction_scale)

        if self.server_debug_writer is not None:
            self._write_server_debug(
                context,
                clients,
                update_norm,
                correction_norm,
                correction_scale,
                cosine,
                self._buffer_update_norms(mean_updates),
                dual_scale,
            )
        if context == "train" and self.dual_state_debug_file is not None:
            self.dual_state_debug_file.flush()
        return aggregated_state

    def aggregate(self, list_clients):
        return self.aggregate_from_state(
            self.global_model.state_dict(),
            list_clients,
            self.dual_state_scale,
            context="train",
        )

    def aggregate_isp_candidate(self, clients):
        return self.aggregate_from_state(
            self.saved_weights,
            clients,
            self.saved_dual_state_scale,
            context="isp_search",
            cache_correction=True,
        )

    def close_debug_files(self):
        if self.dual_state_debug_file is not None:
            self.dual_state_debug_file.close()
        if self.server_debug_file is not None:
            self.server_debug_file.close()
