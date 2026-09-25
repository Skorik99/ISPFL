import json

import numpy as np

from .clients import ISPFedNovaClient, ISPTextClient
from .isp_gp import ISPGP
from .proxy_cohort import (
    proxy_label_distribution,
    select_population_matching_cohort,
)
from .servers import ISPFedCBSServer, ISPFedNovaServer, ISPTextServer


class ISPGPFedNova(ISPGP):
    def _init_server(self, cfg):
        self.server = ISPFedNovaServer(cfg)
        self.server.amount_classes = int(cfg.training_params.num_classes)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = ISPFedNovaClient
        self.client_kwargs["client_cls"] = self.client_cls

    def parse_communication_content(self, client_result):
        super().parse_communication_content(client_result)
        self.server.client_local_steps[client_result["rank"]] = int(
            client_result["local_steps"]
        )

    def calculate_ts(self):
        sizes = np.asarray(
            [self.client_sizes[client] for client in self.list_clients], dtype=float
        )
        fedavg_weights = sizes / sizes.sum()
        local_steps = np.asarray(
            [max(1, self.server.client_local_steps[client]) for client in self.list_clients],
            dtype=float,
        )
        effective_steps = np.sum(fedavg_weights * local_steps)
        return (fedavg_weights * effective_steps / local_steps).tolist()


class ISPGPFedCBS(ISPGP):
    def __init__(self, lambda_, **kwargs):
        self.lambda_ = float(lambda_)
        super().__init__(**kwargs)

    def _init_server(self, cfg):
        self.server = ISPFedCBSServer(cfg, self.lambda_, self.df)

    def _sample_candidate_clients(self, auxiliary_clients, m):
        return self.server.sample_candidate(auxiliary_clients, m, self.rng)


class ISPGPShakespeare(ISPGP):
    def _init_server(self, cfg):
        self.server = ISPTextServer(cfg)
        self.server.amount_classes = int(cfg.training_params.num_classes)

    def _init_client_cls(self):
        self.client_cls = ISPTextClient
        self.client_args = [self.cfg, self.df]
        self.client_kwargs = {
            "client_cls": self.client_cls,
            "pipe": None,
            "rank": None,
            "attack_type": None,
        }

    def _proxy_cohort(self, round_index):
        temperature = float(self.proxy_cfg["temperature"])
        distributions = np.stack(
            [
                proxy_label_distribution(update, temperature)
                for update in self.proxy_updates
            ]
        )
        cohort = select_population_matching_cohort(distributions)
        self.artifacts.append(
            "proxy_cohorts.csv",
            {
                "audit": self.audit_index,
                "round": round_index,
                "clients": json.dumps(cohort.clients),
                "size": len(cohort.clients),
                "selection_rule": "population_proxy_l1",
                "discrepancy": cohort.discrepancy,
                "mean_proxy_distribution": json.dumps(
                    cohort.mean_distribution.tolist()
                ),
                "population_proxy_distribution": json.dumps(
                    cohort.population_distribution.tolist()
                ),
                "uniform_prior_clients": int(
                    np.sum(self.proxy_update_rounds[cohort.clients] < 0)
                ),
            },
        )
        return cohort.clients
