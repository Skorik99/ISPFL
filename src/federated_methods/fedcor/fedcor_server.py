from ..fedavg.fedavg_server import FedAvgServer
import random as rand
import time
import os

from .GPR import Kernel_GPR
from .GPR import Poly_Kernel


class FedCorServer(FedAvgServer):
    def __init__(self, cfg, warmup):
        super().__init__(cfg)

        self.gpr = None
        self.ts = None
        self.gt_global_losses = []
        self.warmup = warmup
        self.local_rnd = rand.Random()

    def select_clients_to_train(self, num_clients_subset, server_sampling=False):
        if self.gpr is None:
            self.gpr = Kernel_GPR(
                num_users=self.amount_of_clients,
                loss_type="MML",
                reusable_history_length=500,
                gamma=0.99,
                device="cpu",
                dimension=15,
                kernel=Poly_Kernel,
                order=1,
                Normalize=0,
            )

        if (self.cur_round <= self.warmup):
            return self.local_rnd.sample(
                list(range(self.amount_of_clients)), num_clients_subset
            )

        if num_clients_subset == self.amount_of_clients:
            return list(range(self.amount_of_clients))

        selected_clients = self.gpr.Select_Clients(
            number=num_clients_subset,
            epsilon=0,
            weights=self.ts,
            Dynamic=False,
            Dynamic_TH=0.0,
        )

        return selected_clients
