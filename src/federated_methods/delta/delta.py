from ..fedavg.fedavg import FedAvg
from ..aggregation import aggregate_weighted_updates
from .delta_client import DeltaClient
from .delta_server import DeltaServer


class Delta(FedAvg):
    def __init__(self, num_clients_subset, alpha_1, alpha_2):
        super().__init__(num_clients_subset)
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = DeltaClient
        self.client_kwargs["client_cls"] = self.client_cls

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)

    def _init_server(self, cfg):
        self.server = DeltaServer(cfg, self.alpha_1, self.alpha_2)

    def aggregate(self):
        self.server.client_probs = self.server.update_probs(self.list_clients)
        print(f"\nClients probabilities:")
        for i, prob in enumerate(self.server.client_probs):
            print(f'{i} : {prob}')

        return aggregate_weighted_updates(
            self.server.global_model.state_dict(),
            self.server.client_gradients,
            self.list_clients,
            self.ts,
            self.server.device,
        )
