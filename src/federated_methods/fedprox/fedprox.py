from ..fedavg.fedavg import FedAvg
from .fedprox_client import FedProxClient


class FedProx(FedAvg):
    def __init__(self, num_clients_subset, mu):
        super().__init__(num_clients_subset)
        self.mu = mu

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedProxClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend([self.mu])
