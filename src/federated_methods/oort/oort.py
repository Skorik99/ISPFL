from ..fedavg.fedavg import FedAvg
from ..aggregation import aggregate_weighted_updates
from .oort_server import OortServer
from .oort_client import OortClient


class Oort(FedAvg):
    def __init__(self, num_clients_subset, slow_clients_count=10):
        super().__init__(num_clients_subset)
        self.slow_clients_count = slow_clients_count

    def _init_server(self, cfg):
        self.server = OortServer(cfg, self.slow_clients_count)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = OortClient
        self.client_kwargs["client_cls"] = self.client_cls

    def get_communication_content(self, rank):
        if getattr(self, "_cpu_model_state_round", None) != self.cur_round:
            self._cpu_model_state_round = self.cur_round
            self._cpu_model_state = {
                name: tensor.detach().cpu()
                for name, tensor in self.server.global_model.state_dict().items()
            }

        return {
            "update_model": self._cpu_model_state,
            "attack_type": (
                self.client_map_round[rank],
                self.attack_configs[self.client_map_round[rank]],
            ),
            "need_train": rank in self.list_clients,
        }

    def aggregate(self):
        return aggregate_weighted_updates(
            self.server.global_model.state_dict(),
            self.server.client_gradients,
            self.list_clients,
            self.ts,
            self.server.device,
        )
