import time

from federated_methods.base.base import Base
from federated_methods.fedavg.fedavg import FedAvg
from hydra.utils import instantiate

from utils.attack_utils import set_client_map_round

from .text_fedavg_client import TextFedAvgClient
from .text_fedavg_server import TextFedAvgServer


class TextFedAvg(FedAvg):
    def _init_client_cls(self):
        Base._init_client_cls(self)
        self.client_cls = TextFedAvgClient
        self.client_kwargs["client_cls"] = self.client_cls

    def _init_server(self, cfg):
        self.server = TextFedAvgServer(cfg)
        self.server.amount_classes = cfg.training_params.num_classes
