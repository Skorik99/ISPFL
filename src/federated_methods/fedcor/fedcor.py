from ..fedavg.fedavg import FedAvg
from ..aggregation import aggregate_weighted_updates
from .fedcor_client import FedCorClient
from .fedcor_server import FedCorServer
import random as rand
import numpy as np


class FedCor(FedAvg):
    def __init__(self, num_clients_subset, warmup):
        super().__init__(num_clients_subset)
        self.warmup = warmup

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedCorClient
        self.client_kwargs["client_cls"] = self.client_cls

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)

        clients_df_len = [
            len(self.df[self.df["client"] == i]) for i in range(self.amount_of_clients)
        ]
        self.server.ts = np.array([client_len / sum(clients_df_len) for client_len in clients_df_len])

    def _init_server(self, cfg):
        self.server = FedCorServer(cfg, self.warmup)

    def aggregate(self):
        val_losses = [metrics[1] for metrics in self.server.server_metrics]
        self.server.gt_global_losses.append(val_losses)

        if self.cur_round >= 1:
            if self.cur_round <= self.warmup:  # warm-up
                self.server.gpr.Update_Training_Data(
                    [
                        np.arange(self.amount_of_clients),
                    ],
                    [
                        np.array(self.server.gt_global_losses[-1]) - np.array(self.server.gt_global_losses[-2]),
                    ],
                    epoch=self.cur_round,
                )

                if self.cur_round == self.warmup:
                    print("Training GPR")
                    self.server.gpr.Train(
                        lr=1e-2,
                        llr=0.01,
                        max_epoches=1000,
                        schedule_lr=False,
                        update_mean=True,
                        verbose=1,
                    )

            elif (
                self.cur_round > self.warmup and self.cur_round % 50 == 0
            ):  # normal and optimization round
                self.server.gpr.Reset_Discount()
                print("Training with Random Selection For GPR Training:")
                
                # We use current round, don`t call unnecessary communication...
                gpr_loss = self.server.gt_global_losses[-1] 
                self.server.gpr.Update_Training_Data(
                    [
                        np.arange(self.amount_of_clients),
                    ],
                    [
                        np.array(gpr_loss) - np.array(self.server.gt_global_losses[-2]),
                    ],
                    epoch=self.cur_round,
                )
                print("Training GPR")
                self.server.gpr.Train(
                    lr=1e-2,
                    llr=0.01,
                    max_epoches=100,
                    schedule_lr=False,
                    update_mean=True,
                    verbose=1,
                )
            else:
                self.server.gpr.Update_Discount(self.list_clients, 0.9)

        return aggregate_weighted_updates(
            self.server.global_model.state_dict(),
            self.server.client_gradients,
            self.list_clients,
            self.ts,
            self.server.device,
        )
