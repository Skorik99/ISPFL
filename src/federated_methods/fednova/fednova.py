import warnings

import numpy as np

from ..fedavg.fedavg import FedAvg
from .fednova_client import FedNovaClient


class FedNova(FedAvg):
    def __init__(self, num_clients_subset):
        super().__init__(num_clients_subset)
        self.client_local_steps = {}

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)
        if "SGD" not in str(cfg.optimizer._target_):
            warnings.warn(
                (
                    "FedNova is designed for use with the SGD optimizer. "
                    f"Current optimizer: {cfg.optimizer._target_}"
                ),
                UserWarning,
            )

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedNovaClient
        self.client_kwargs["client_cls"] = self.client_cls

    def parse_communication_content(self, client_result):
        super().parse_communication_content(client_result)
        if client_result["rank"] in self.list_clients:
            self.client_local_steps[client_result["rank"]] = int(
                client_result["local_steps"]
            )

    def calculate_ts(self):
        return self.calculate_ts_for_clients(self.list_clients)

    def calculate_ts_for_clients(self, clients):
        client_data_sizes = np.array(
            [len(self.df[self.df["client"] == rank]) for rank in clients],
            dtype=np.float64,
        )
        fedavg_weights = client_data_sizes / client_data_sizes.sum()

        local_steps = np.array(
            [max(1, self.client_local_steps[rank]) for rank in clients],
            dtype=np.float64,
        )
        effective_tau = np.sum(fedavg_weights * local_steps)

        return (fedavg_weights * effective_tau / local_steps).tolist()
