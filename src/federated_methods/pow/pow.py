from ..fedavg.fedavg import FedAvg
from ..aggregation import aggregate_weighted_updates
from .pow_client import PowClient
from .pow_server import PowServer

class Pow(FedAvg):
    def __init__(self, num_clients_subset, candidate_set_size):
        super().__init__(num_clients_subset)

        # candidate_set_size is 'd' in paper
        self.candidate_set_size = candidate_set_size

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = PowClient
        self.client_kwargs["client_cls"] = self.client_cls
    
    def _init_server(self, cfg):
        self.server = PowServer(cfg, self.candidate_set_size, self.df)

    def parse_communication_content(self, client_result):
        super().parse_communication_content(client_result)
        if client_result["rank"] in self.list_clients:
            self.server.clients_losses[client_result["rank"]] = client_result["client_loss"]

    def aggregate(self):
        return aggregate_weighted_updates(
            self.server.global_model.state_dict(),
            self.server.client_gradients,
            self.list_clients,
            self.ts,
            self.server.device,
        )
