from ..fedavg.fedavg_client import FedAvgClient

class FedCorClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
