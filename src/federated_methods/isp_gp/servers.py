from collections.abc import Mapping

import numpy as np
import torch

from ..fedavg.fedavg_server import FedAvgServer
from ..fedcbs.fedcbs_server import FedCBSServer
from ..text_base.text_fedavg_server import TextFedAvgServer


class ISPServerMixin:
    def save_best_model(self, round_index):
        all_metrics = self.server_metrics
        self.server_metrics = [all_metrics[client] for client in self.list_clients]
        try:
            super().save_best_model(round_index)
        finally:
            self.server_metrics = all_metrics

    def aggregate_candidate(
        self,
        base_state: Mapping[str, torch.Tensor],
        clients: list[int],
        client_sizes: Mapping[int, int],
    ) -> dict[str, torch.Tensor]:
        total_size = sum(client_sizes[client] for client in clients)
        state = {
            key: value.detach().cpu().clone() for key, value in base_state.items()
        }
        for client in clients:
            weight = client_sizes[client] / total_size
            for key, update in self.client_gradients[client].items():
                update = update.detach().cpu()
                if state[key].is_floating_point():
                    state[key].add_(update, alpha=weight)
                else:
                    state[key] = state[key] + update * weight
        return state


class ISPServer(ISPServerMixin, FedAvgServer):
    pass


class ISPFedNovaServer(ISPServer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.client_local_steps = {}

    def aggregate_candidate(self, base_state, clients, client_sizes):
        sizes = np.asarray([client_sizes[client] for client in clients], dtype=float)
        fedavg_weights = sizes / sizes.sum()
        local_steps = np.asarray(
            [max(1, self.client_local_steps[client]) for client in clients],
            dtype=float,
        )
        effective_steps = np.sum(fedavg_weights * local_steps)
        weights = fedavg_weights * effective_steps / local_steps
        state = {
            key: value.detach().cpu().clone() for key, value in base_state.items()
        }
        for client, weight in zip(clients, weights):
            for key, update in self.client_gradients[client].items():
                update = update.detach().cpu()
                if state[key].is_floating_point():
                    state[key].add_(update, alpha=float(weight))
                else:
                    state[key] = state[key] + update * float(weight)
        return state


class ISPFedCBSServer(ISPServerMixin, FedCBSServer):
    def sample_candidate(self, candidates, size, rng):
        selected = []
        remaining = list(candidates)
        betas = np.arange(1, size + 1)
        selected_data_count = 0.0
        selected_qcid_sum = 0.0

        def candidate_qcids():
            counts = np.asarray(
                [self.client_data_count[client] for client in remaining], dtype=float
            )
            diagonal = self.qcid_mtr[remaining, remaining]
            cross = (
                self.qcid_mtr[np.ix_(selected, remaining)].sum(axis=0)
                if selected
                else 0.0
            )
            return (
                selected_qcid_sum + 2.0 * cross + diagonal
            ) / (selected_data_count + counts) ** 2 - 1.0 / self.amount_classes

        for index in range(size):
            if index == 0:
                qcids = candidate_qcids()
                probabilities = np.asarray(
                    [
                        1.0 / qcid**betas[0]
                        + self.lambda_
                        * np.sqrt(
                            3.0
                            * np.log(self.cur_round + 1)
                            / (2.0 * self.selection_counter[client])
                        )
                        for client, qcid in zip(remaining, qcids)
                    ]
                )
            elif index == 1:
                current_qcid = (
                    selected_qcid_sum / selected_data_count**2
                    - 1.0 / self.amount_classes
                )
                qcids = candidate_qcids()
                probabilities = np.asarray(
                    [
                        1.0
                        / qcid**betas[1]
                        / (
                            1.0 / current_qcid**betas[0]
                            + self.lambda_
                            * np.sqrt(
                                3.0
                                * np.log(self.cur_round + 1)
                                / (2.0 * self.selection_counter[client])
                            )
                        )
                        for client, qcid in zip(remaining, qcids)
                    ]
                )
            else:
                current_qcid = (
                    selected_qcid_sum / selected_data_count**2
                    - 1.0 / self.amount_classes
                )
                qcids = candidate_qcids()
                probabilities = (
                    (current_qcid / qcids) ** betas[index - 2] / qcids
                )
            probabilities = np.where(np.isfinite(probabilities), probabilities, 1e-6)
            probabilities[probabilities <= 0.0] = 1e-6
            probabilities /= probabilities.sum()
            client = int(rng.choice(remaining, p=probabilities))
            selected_qcid_sum += (
                2.0 * self.qcid_mtr[selected, client].sum()
                + self.qcid_mtr[client, client]
            )
            selected_data_count += self.client_data_count[client]
            selected.append(client)
            remaining.remove(client)
        return sorted(selected)


class ISPTextServer(ISPServerMixin, TextFedAvgServer):
    pass
