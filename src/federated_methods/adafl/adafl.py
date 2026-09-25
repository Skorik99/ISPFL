import time
from utils.attack_utils import (
    set_client_map_round,
)

from types import MethodType

from ..fedavg.fedavg import FedAvg
from ..fedcbs.fedcbs import FedCBS
from ..delta.delta import Delta
from ..pow.pow import Pow
from ..fedcor.fedcor import FedCor


class AdaFL:
    def __init__(
        self,
        base_method,
        wp_amount_cl,
        max_amount_cl,
        step_increase,
        warmup_rounds,
        local_epoch_on_wp,
        args,
    ):
        self.base_method = base_method
        self.wp_amount_cl = wp_amount_cl
        self.max_amount_cl = max_amount_cl
        self.step_increase = step_increase
        self.warmup_rounds = warmup_rounds
        self.local_epoch_on_wp = local_epoch_on_wp
        self.args = args

    def _init_federated(self, cfg, df):
        if self.base_method == "fedavg":
            name_method = "FedAvg"
            metaclass = type(FedAvg)
            base_class = metaclass(f"LightClass_{name_method}", (FedAvg,), {})

        elif self.base_method == "fedcbs":
            name_method = "FedCBS"
            metaclass = type(FedCBS)
            base_class = metaclass(f"LightClass_{name_method}", (FedCBS,), {})

        elif self.base_method == "delta":
            name_method = "Delta"
            metaclass = type(Delta)
            base_class = metaclass(f"LightClass_{name_method}", (Delta,), {})

        elif self.base_method == "pow":
            name_method = "Pow"
            metaclass = type(Pow)
            base_class = metaclass(f"LightClass_{name_method}", (Pow,), {})

        elif self.base_method == "fedcor":
            name_method = "Fedcor"
            metaclass = type(FedCor)
            base_class = metaclass(f"LightClass_{name_method}", (FedCor,), {})

        else:
            raise NotImplementedError(
                f"{self.base_method} not available to AdaFL version"
            )

        adafl_method = base_class(**self.args)
        adafl_method._init_federated(cfg, df)

        adafl_method.name_method = name_method

        adafl_method.base_method = self.base_method
        adafl_method.wp_amount_cl = self.wp_amount_cl
        adafl_method.max_amount_cl = self.max_amount_cl
        adafl_method.step_increase = self.step_increase
        adafl_method.warmup_rounds = self.warmup_rounds
        adafl_method.local_epoch_on_wp = self.local_epoch_on_wp
        adafl_method.amount_of_clients = cfg.federated_params.amount_of_clients

        adafl_method.map_amount_clients = self.calculate_client_amounts(
            adafl_method.rounds
        )
        print(f"Map amount of clients:\n{adafl_method.map_amount_clients}\n")

        # Set parametrs to server
        adafl_method.server.amount_of_clients = adafl_method.amount_of_clients

        adafl_method.begin_train = MethodType(AdaFL.begin_train, adafl_method)
        adafl_method.get_amount_clients = MethodType(
            AdaFL.get_amount_clients, adafl_method
        )

        # Change methods for setting local_epoch on wp
        adafl_method.full_method_get_communication_content = MethodType(
            base_class.get_communication_content,
            adafl_method,
        )
        adafl_method.get_communication_content = MethodType(
            AdaFL.get_communication_content,
            adafl_method,
        )
        adafl_method.client_cls.full_method_create_pipe_commands = (
            adafl_method.client_cls.create_pipe_commands
        )
        adafl_method.client_cls.create_pipe_commands = AdaFL.client_create_pipe_commands
        adafl_method.client_cls.train_fn = AdaFL.client_train_fn
        adafl_method.client_cls.set_local_epoch = AdaFL.client_set_local_epoch

        return adafl_method

    def calculate_client_amounts(self, rounds):
        client_amounts = {}
        for t in range(rounds):
            if t <= self.warmup_rounds:
                m_t = self.wp_amount_cl
            else:
                m_t = round(
                    min(
                        self.wp_amount_cl
                        + (t - self.warmup_rounds) / self.step_increase,
                        self.max_amount_cl,
                    )
                )
            client_amounts[t] = m_t
        return client_amounts

    def get_amount_clients(self):
        self.num_clients_subset = self.map_amount_clients[self.cur_round]
        return self.num_clients_subset

    def begin_train(self):
        self.manager.create_clients(
            self.client_args, self.client_kwargs, self.client_attack_map
        )
        self.clients_loader = self.manager.batches

        print()
        print(f"AdaFL version of {self.name_method}")
        print(f"Number of warmup rounds is {self.warmup_rounds}")

        for cur_round in range(self.rounds):
            print(f"\nRound number: {cur_round}")
            begin_round_time = time.time()
            self.cur_round = cur_round
            self.server.cur_round = cur_round

            self.num_clients_subset = self.get_amount_clients()
            self.list_clients = self.server.select_clients_to_train(
                self.num_clients_subset
            )
            self.list_clients.sort()
            print(f"Clients on this communication: {self.list_clients}\n")
            print(
                f"Amount of clients on this communication: {len(self.list_clients)}\n"
            )

            print("\nTraining started\n")
            self.client_map_round = set_client_map_round(
                self.client_attack_map,
                self.attack_rounds,
                self.attack_scheme,
                cur_round,
            )

            self.train_round()

            self.server.test_global_model()
            self.server.save_best_model(cur_round)

            self.ts = self.calculate_ts()
            print(f"Client weights for aggregation on this communication {self.ts}")

            aggregated_weights = self.aggregate()
            self.server.global_model.load_state_dict(aggregated_weights)

            print(f"Round time: {time.time() - begin_round_time}", flush=True)

        print("Shutdown clients, federated learning end", flush=True)
        self.manager.stop_train()

    #
    #
    # Local epoch on warmup changing:
    #
    #

    def get_communication_content(self, rank):
        content = self.full_method_get_communication_content(rank)
        content["adafl_method_local_epoch"] = (
            self.local_epoch_on_wp
            if (self.cur_round < self.warmup_rounds)
            else self.cfg.federated_params.round_epochs
        )
        return content

    def client_create_pipe_commands(self):
        pipe_commands_map = self.full_method_create_pipe_commands()
        pipe_commands_map["adafl_method_local_epoch"] = self.set_local_epoch
        return pipe_commands_map

    def client_set_local_epoch(self, local_epoch):
        self.local_epoch = local_epoch

    def client_train_fn(self):
        # Diff with original client in 'range(self.local_epoch)'
        self.model.train()

        for _ in range(self.local_epoch):

            for batch in self.train_loader:
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inp)

                loss = self.get_loss_value(outputs, targets)

                loss.backward()

                self.optimizer.step()

                inp = input[0].to("cpu")
                targets = targets.to("cpu")
