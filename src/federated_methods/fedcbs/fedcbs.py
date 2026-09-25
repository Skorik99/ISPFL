from ..fedavg.fedavg import FedAvg
from ..aggregation import aggregate_weighted_updates
from .fedcbs_server import FedCBSServer


class FedCBS(FedAvg):
    def __init__(self, num_clients_subset, lambda_):
        super().__init__(num_clients_subset)
        self.lambda_ = lambda_

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)

    def _init_server(self, cfg):
        self.server = FedCBSServer(cfg, self.lambda_, self.df)

    def aggregate(self):
        return aggregate_weighted_updates(
            self.server.global_model.state_dict(),
            self.server.client_gradients,
            self.list_clients,
            self.ts,
            self.server.device,
        )
