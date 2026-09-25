import time
import warnings

from hydra.utils import instantiate

from utils.attack_utils import set_client_map_round

from ..fedavg.fedavg import FedAvg
from .feddyn_client import FedDynClient
from .feddyn_server import FedDynServer


class FedDyn(FedAvg):
    def __init__(
        self,
        num_clients_subset,
        alpha,
        debug_dual_state=False,
        dual_state_decay_start_round=200,
        dual_state_min_scale=0.1,
        server_correction_max_ratio=0.85,
    ):
        super().__init__(num_clients_subset)
        self.alpha = float(alpha)
        self.debug_dual_state = bool(debug_dual_state)
        self.dual_state_decay_start_round = int(dual_state_decay_start_round)
        self.dual_state_min_scale = float(dual_state_min_scale)
        self.server_correction_max_ratio = float(server_correction_max_ratio)

    def _get_dual_state_scale(self, cur_round):
        return max(
            self.dual_state_min_scale,
            min(1.0, self.dual_state_decay_start_round / (cur_round + 1)),
        )

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)
        if "SGD" not in str(cfg.optimizer._target_):
            warnings.warn(
                (
                    "FedDyn reference code uses SGD-style optimization. "
                    f"Current optimizer: {cfg.optimizer._target_}"
                ),
                UserWarning,
            )

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedDynClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend([self.alpha])

    def _init_server(self, cfg):
        self.server = FedDynServer(
            cfg,
            self.alpha,
            self.server_correction_max_ratio,
        )
        self.server.amount_classes = len(self.df["target"].unique())

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        content["local_dual_state"] = self.server.get_client_dual_state(rank)
        content["dual_state_scale"] = self.server.dual_state_scale
        return content

    def aggregate(self):
        return self.server.aggregate(self.list_clients)

    def begin_train(self):
        self.manager.create_clients(
            self.client_args, self.client_kwargs, self.client_attack_map
        )
        self.clients_loader = self.manager.batches
        self.server.global_model = instantiate(self.cfg.models[0])
        self.server.initialize_dual_states(self.server.global_model)

        for cur_round in range(self.rounds):
            print(f"\nRound number: {cur_round}")
            begin_round_time = time.time()
            self.cur_round = cur_round
            self.server.cur_round = cur_round
            self.server.dual_state_scale = self._get_dual_state_scale(cur_round)

            print("\nTraining started\n")

            self.client_map_round = set_client_map_round(
                self.client_attack_map, self.attack_rounds, self.attack_scheme, cur_round
            )

            self.list_clients = self.server.select_clients_to_train(
                self.num_clients_subset
            )
            self.list_clients.sort()
            self.server.list_clients = self.list_clients
            print(f"Clients on this communication: {self.list_clients}")
            print(f"Amount of clients on this communication: {len(self.list_clients)}\n")

            self.train_round()

            self.server.test_global_model()
            self.server.save_best_model(cur_round)

            aggregated_weights = self.aggregate()
            self.server.global_model.load_state_dict(aggregated_weights)

            print(f"Round time: {time.time() - begin_round_time}", flush=True)

        print("Shutdown clients, federated learning end", flush=True)
        self.manager.stop_train()
        self.server.close_debug_files()
