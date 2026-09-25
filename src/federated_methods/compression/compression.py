from ..fedavg.fedavg import FedAvg
from .compression_client import CompressionClient


class Compression(FedAvg):
    def __init__(self, num_clients_subset, compression_type, compression_k_percent):
        super().__init__(num_clients_subset)
        self.num_clients_subset = num_clients_subset
        self.compression_type = compression_type
        self.compression_k_percent = compression_k_percent

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = CompressionClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend([self.compression_type, self.compression_k_percent])
