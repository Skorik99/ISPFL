import time
import copy
import random as rand
import torch
import numpy as np
import pandas as pd

from types import MethodType
from hydra.utils import instantiate

from utils.losses import get_loss
from utils.data_utils import get_dataset_loader
from sklearn.model_selection import train_test_split
from utils.data_utils import read_dataframe_from_cfg, print_df_distribution
from utils.cs_statistics import ClientSelectionStatisticsCollector
from utils.attack_utils import (
    set_client_map_round,
)

from ..fedavg.fedavg import FedAvg
from federated_methods.text_base.text_fedavg import TextFedAvg
from ..fedcbs.fedcbs import FedCBS
from ..delta.delta import Delta
from ..pow.pow import Pow
from ..fedcor.fedcor import FedCor
from ..oort.oort import Oort
from ..compression.compression import Compression
from ..fedprox.fedprox import FedProx
from ..fednova.fednova import FedNova
from ..feddyn.feddyn import FedDyn


class ISP:
    """Implementation-level ISP controller corresponding to Algorithms 3--6."""

    def __init__(
        self,
        base_method,
        warmup_rounds,
        step_find_optimal,
        num_samples,
        sample_step,
        num_clients_momentum,
        window_sz_loss,
        local_epoch_on_wp,
        full_comm_client_amount,
        full_comm_amount_cl_multiplayer,
        use_stale_updates,
        debug_dual_state,
        trust_available,
        corrupted_trust,
        ema_use,
        args,
    ):
        self.base_method = base_method
        self.warmup_rounds = warmup_rounds
        self.step_find_optimal = step_find_optimal
        self.num_samples = num_samples
        self.sample_step = sample_step
        self.num_clients_momentum = num_clients_momentum
        self.window_sz_loss = window_sz_loss
        self.local_epoch_on_wp = local_epoch_on_wp
        self.full_comm_client_amount = full_comm_client_amount
        self.full_comm_amount_cl_multiplayer = full_comm_amount_cl_multiplayer
        self.use_stale_updates = use_stale_updates
        self.debug_dual_state = debug_dual_state
        assert use_stale_updates or base_method in {"fedavg", "feddyn", "fedprox", "fednova"}, (
            "Fresh update sampling supports only FedAvg, FedDyn, FedProx, and FedNova"
        )
        self.trust_available = trust_available
        self.corrupted_trust = corrupted_trust
        self.ema_use = ema_use
        self.args = args

    def _init_federated(self, cfg, df):
        if self.base_method == "fedavg":
            # Use text-specific FedAvg for Shakespeare datasets
            base_cls = TextFedAvg if "shakespeare" in cfg.dataset.data_name else FedAvg
            name_method = base_cls.__name__
            metaclass = type(base_cls)
            isp_class = metaclass(f"ISPClass_{name_method}", (base_cls,), {})

        elif self.base_method == "fedcbs":
            name_method = "FedCBS"
            metaclass = type(FedCBS)
            isp_class = metaclass(f"ISPClass_{name_method}", (FedCBS,), {})

        elif self.base_method == "delta":
            name_method = "Delta"
            metaclass = type(Delta)
            isp_class = metaclass(f"ISPClass_{name_method}", (Delta,), {})

        elif self.base_method == "pow":
            name_method = "Pow"
            metaclass = type(Pow)
            isp_class = metaclass(f"ISPClass_{name_method}", (Pow,), {})

        elif self.base_method == "fedcor":
            name_method = "Fedcor"
            metaclass = type(FedCor)
            isp_class = metaclass(f"ISPClass_{name_method}", (FedCor,), {})

        elif self.base_method == "compression":
            name_method = "Compression"
            metaclass = type(Compression)
            isp_class = metaclass(f"ISPClass_{name_method}", (Compression,), {})

        elif self.base_method == "fedprox":
            name_method = "FedProx"
            metaclass = type(FedProx)
            isp_class = metaclass(f"ISPClass_{name_method}", (FedProx,), {})

        elif self.base_method == "feddyn":
            name_method = "FedDyn"
            metaclass = type(FedDyn)
            isp_class = metaclass(f"ISPClass_{name_method}", (FedDyn,), {})

        elif self.base_method == "fednova":
            name_method = "FedNova"
            metaclass = type(FedNova)
            isp_class = metaclass(f"ISPClass_{name_method}", (FedNova,), {})

        elif self.base_method == "oort":
            name_method = "Oort"
            metaclass = type(Oort)
            isp_class = metaclass(f"ISPClass_{name_method}", (Oort,), {})

        else:
            raise NotImplementedError(
                f"{self.base_method} not available to ISP version"
            )

        isp_method = isp_class(**self.args)
        if self.base_method == "feddyn":
            cfg.federated_method.debug_dual_state = isp_method.debug_dual_state
        isp_method._init_federated(cfg, df)

        isp_method.name_method = name_method
        isp_method.warmup_rounds = self.warmup_rounds
        isp_method.step_find_optimal = self.step_find_optimal
        isp_method.num_samples = self.num_samples
        isp_method.sample_step = self.sample_step
        isp_method.window_sz_loss = self.window_sz_loss
        isp_method.local_epoch_on_wp = self.local_epoch_on_wp
        isp_method.amount_of_clients = cfg.federated_params.amount_of_clients
        isp_method.initial_num_clients_subset = int(isp_method.num_clients_subset)
        isp_method.optimal_amount_clients = int(isp_method.initial_num_clients_subset)
        isp_method.full_comm_amount_cl_multiplayer = (
            self.full_comm_amount_cl_multiplayer
        )
        isp_method.full_comm_client_amount = self.full_comm_client_amount
        isp_method.current_full_comm_client_amount = (
            ISP.resolve_full_comm_client_amount(
                isp_method, isp_method.optimal_amount_clients
            )
        )
        isp_method.borders_of_clients = [
            1,
            isp_method.current_full_comm_client_amount + 1,
        ]
        isp_method.find_optimal_rounds = [
            i
            for i in range(
                isp_method.warmup_rounds,
                isp_method.rounds,
                isp_method.step_find_optimal,
            )
        ]
        print(f"Find optimal rounds at {isp_method.find_optimal_rounds}\n")
        isp_method.wp_amount_cl_history = []

        # Set parametrs to server
        isp_method.server.amount_of_clients = isp_method.amount_of_clients
        isp_method.server.df = df
        isp_method.server.use_stale_updates = self.use_stale_updates
        isp_method.server.full_round_clients = []

        if self.trust_available:
            trust_df = read_dataframe_from_cfg(cfg, "train_directories", "trust_df")
            isp_method.server.trust_df = trust_df
        else:
            # Trust df = Validation client datasets
            # We don't pass models to clients in our code to avoid complex programming.
            # We simply use client validation as a trust dataset,
            # thus unifying the ISP pipeline from the technical implementation side.
            val_client_df = pd.DataFrame()
            n_classes = cfg.training_params.num_classes
            train_val_prop = cfg.federated_params.client_train_val_prop

            is_text = "shakespeare" in cfg.dataset.data_name

            for cl_rank in range(isp_method.amount_of_clients):
                cur_train_val_prop = train_val_prop
                client_df = df[df["client"] == cl_rank]

                if is_text:
                    # For text, targets are sequences; use a simple random split
                    _, client_val = train_test_split(
                        client_df,
                        test_size=cur_train_val_prop,
                        random_state=cfg.random_state,
                        shuffle=True,
                    )
                else:
                    minor_classes_ids = (
                        client_df["target"]
                        .value_counts()[client_df["target"].value_counts() < 2]
                        .index
                    )
                    major_classes_df = client_df[
                        ~client_df["target"].isin(minor_classes_ids)
                    ]

                    if cur_train_val_prop * len(major_classes_df) < n_classes:
                        cur_train_val_prop = (
                            1 / major_classes_df["target"].value_counts().min()
                        )

                    train_part, client_val = train_test_split(
                        major_classes_df,
                        test_size=cur_train_val_prop,
                        stratify=major_classes_df["target"],
                        random_state=cfg.random_state,
                    )
                val_client_df = pd.concat([val_client_df, client_val])

            if "shakespeare" not in cfg.dataset.data_name:
                print("Trust Distribution:")
                print_df_distribution(
                    val_client_df, n_classes, isp_method.amount_of_clients
                )
                print("\n\n")

            trust_df = val_client_df.copy()
            isp_method.server.trust_df = val_client_df

        mode = "valid"
        if self.trust_available and self.corrupted_trust:
            mode = "trust_corrupted"

        isp_method.server.eval_loader = get_dataset_loader(
            isp_method.server.trust_df,
            isp_method.server.cfg,
            drop_last=False,
            mode=mode,
        )

        isp_method.server.criterion = get_loss(
            loss_cfg=isp_method.server.cfg.loss,
            device=isp_method.server.device,
            df=isp_method.server.trust_df,
            init_pos_weight=None,
        )
        isp_method.server.ema_use = self.ema_use
        isp_method.server.sample_step = self.sample_step
        isp_method.server.num_samples = self.num_samples
        isp_method.server.window_sz_loss = self.step_find_optimal
        isp_method.server.trust_losses = []
        isp_method.server.clients_history = []
        isp_method.server.saved_weights = None
        isp_method.server.wp_mean_amount_cl = None
        isp_method.server.wp_mean_amount_cl_loss = None
        isp_method.server.cnt_solve_optimal_task = 0
        isp_method.server.num_clients_momentum = self.num_clients_momentum
        isp_method.server.num_classes = (
            trust_df["target"].nunique()
            if "shakespeare" not in cfg.dataset.data_name
            else cfg.training_params.num_classes
        )
        print(f"Num classes on task is {isp_method.server.num_classes}")
        isp_method.sampling_statistics = ClientSelectionStatisticsCollector(cfg)

        isp_method.collect_sampling_statistics_for_find_optimal = MethodType(
            ISP.collect_sampling_statistics_for_find_optimal, isp_method
        )
        isp_method.resolve_full_comm_client_amount = MethodType(
            ISP.resolve_full_comm_client_amount, isp_method
        )
        isp_method.update_full_comm_budget = MethodType(
            ISP.update_full_comm_budget, isp_method
        )

        isp_method.begin_train = MethodType(ISP.begin_train, isp_method)
        isp_method.get_amount_clients = MethodType(ISP.get_amount_clients, isp_method)

        isp_method.full_method_aggregate = MethodType(
            isp_class.aggregate,
            isp_method,  # Copy default aggregate in another method
        )
        isp_method.aggregate = MethodType(
            ISP.aggregate,
            isp_method,  # ISP aggregate need save weight before aggregate
        )

        isp_method.server.eval_global_model = MethodType(
            ISP.eval_global_model, isp_method.server
        )
        # Text-specific eval for Shakespeare datasets
        if "shakespeare" in cfg.dataset.data_name:
            isp_method.server.eval_global_model = MethodType(
                ISP.text_eval_global_model, isp_method.server
            )

        isp_method.full_method_get_communication_content = MethodType(
            isp_class.get_communication_content,
            isp_method,
        )
        isp_method.get_communication_content = MethodType(
            ISP.get_communication_content,
            isp_method,
        )

        # Change methods for client
        isp_method.client_cls.full_method_create_pipe_commands = (
            isp_method.client_cls.create_pipe_commands
        )
        isp_method.client_cls.create_pipe_commands = ISP.client_create_pipe_commands

        isp_method.client_cls.train_fn = ISP.client_train_fn
        isp_method.client_cls.set_local_epoch = ISP.client_set_local_epoch

        # Change methods for solving optimization task
        isp_method.server.find_optimal = MethodType(ISP.find_optimal, isp_method.server)
        isp_method.server.aggregate_clients = MethodType(
            ISP.aggregate_clients, isp_method.server
        )
        if self.base_method == "fednova":
            isp_method.server.calculate_fednova_ts = (
                isp_method.calculate_ts_for_clients
            )
            isp_method.server.aggregate_clients = MethodType(
                ISP.fednova_aggregate_clients, isp_method.server
            )
        elif self.base_method == "feddyn":
            isp_method.server.aggregate_clients = (
                isp_method.server.aggregate_isp_candidate
            )
        isp_method.server.get_mean_loss = MethodType(
            ISP.get_mean_loss, isp_method.server
        )
        isp_method.server.ema_loss_history = MethodType(
            ISP.ema_loss_history, isp_method.server
        )
        isp_method.server.get_functional_value = MethodType(
            ISP.get_functional_value, isp_method.server
        )
        isp_method.server.choose_optimal = MethodType(
            ISP.choose_optimal, isp_method.server
        )
        isp_method.server.first_positive_functional = MethodType(
            ISP.first_positive_functional, isp_method.server
        )

        return isp_method

    def eval_global_model(self, need_save_loss):
        """Evaluate the server-side objective in Algorithm 6, lines 14--26."""
        self.global_model.to(self.device)
        self.global_model.eval()

        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.trust_df,
        )

        val_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(self.eval_loader):
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                outputs = self.global_model(inp)

                val_loss += self.criterion(outputs, targets).detach().item()

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

                inp = input[0].to("cpu")
                targets = targets.to("cpu")

        trust_loss = val_loss / len(self.eval_loader)

        if need_save_loss:
            self.trust_losses.append(trust_loss)

        return trust_loss

    def text_eval_global_model(self, need_save_loss):
        """Evaluate Algorithm 6's objective with sequence-aware token loss."""
        self.global_model.to(self.device)
        self.global_model.eval()

        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.trust_df,
        )

        total_loss = 0
        total_tokens = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(self.eval_loader):
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                outputs = self.global_model(inp)

                # flatten logits and targets for CE
                if outputs.dim() == 3:
                    outputs_flat = outputs.reshape(-1, outputs.size(-1))
                else:
                    outputs_flat = outputs
                if targets.dim() > 1:
                    targets_flat = targets.reshape(-1)
                else:
                    targets_flat = targets

                loss = self.criterion(outputs_flat, targets_flat)
                total_loss += loss.detach().item()
                total_tokens += (
                    (targets_flat != self.criterion.ignore_index).sum().item()
                )

                fin_targets.extend(targets_flat.tolist())
                fin_outputs.extend(outputs_flat.tolist())

                inp = input[0].to("cpu")
                targets = targets.to("cpu")

        trust_loss = total_loss / len(self.eval_loader)

        if need_save_loss:
            self.trust_losses.append(trust_loss)

        return trust_loss

    def collect_sampling_statistics_for_find_optimal(self):
        if getattr(self, "sampling_statistics", None) is None:
            return
        self.sampling_statistics.collect(self)

    def resolve_full_comm_client_amount(self, reference_num_clients=None):
        if reference_num_clients is None:
            reference_num_clients = getattr(
                self, "optimal_amount_clients", self.initial_num_clients_subset
            )

        if self.full_comm_client_amount is None:
            if self.full_comm_amount_cl_multiplayer < 1:
                raise ValueError(
                    "full_comm_amount_cl_multiplayer must be >= 1 "
                    f"(got {self.full_comm_amount_cl_multiplayer})"
                )
            full_comm_amount = int(
                np.ceil(
                    float(self.full_comm_amount_cl_multiplayer)
                    * float(reference_num_clients)
                )
            )
        else:
            full_comm_amount = int(self.full_comm_client_amount)

        return max(1, min(int(self.amount_of_clients), full_comm_amount))

    def update_full_comm_budget(self, reference_num_clients=None):
        full_comm_amount = self.resolve_full_comm_client_amount(reference_num_clients)
        self.current_full_comm_client_amount = full_comm_amount
        self.borders_of_clients = [1, full_comm_amount + 1]
        return full_comm_amount

    def get_amount_clients(self):
        if self.cur_round < self.warmup_rounds:
            print("Warmup round")
            self.num_clients_subset = self.update_full_comm_budget(
                self.optimal_amount_clients
            )
            return self.num_clients_subset

        if self.cur_round in self.find_optimal_rounds:
            # Algorithm 4, lines 4--6: collect and cache the auxiliary updates.
            print('Full round for "Optimal" method')
            self.num_clients_subset = self.update_full_comm_budget(
                self.optimal_amount_clients
            )
            return self.num_clients_subset

        if self.cur_round - 1 in self.find_optimal_rounds:
            # Evaluate Algorithm 4's direct search from the cached auxiliary round.
            self.collect_sampling_statistics_for_find_optimal()
            self.optimal_amount_clients = self.server.find_optimal(
                self.borders_of_clients
            )
            if self.name_method == "FedDyn":
                self.server.clear_isp_correction_cache()
            self.server.clients_history.append(self.optimal_amount_clients)

        self.num_clients_subset = self.optimal_amount_clients
        return self.num_clients_subset

    def aggregate(self):
        # Preserve x^tau so every Algorithm 5 candidate uses the same base model.
        aggregated_weights = self.server.global_model.state_dict()
        self.server.saved_weights = {
            key: value.detach().clone()
            for key, value in aggregated_weights.items()
        }
        if self.name_method == "FedDyn":
            self.server.saved_dual_state_scale = self.server.dual_state_scale
        if self.name_method != "FedNova":
            self.ts = [1 / len(self.list_clients) for i in self.list_clients]

        return self.full_method_aggregate()

    def begin_train(self):
        self.manager.create_clients(
            self.client_args, self.client_kwargs, self.client_attack_map
        )
        self.clients_loader = self.manager.batches
        self.server.global_model = instantiate(self.cfg.models[0])
        if self.name_method == "FedDyn":
            self.server.initialize_dual_states(self.server.global_model)

        print()
        print(f"ISP version of {self.name_method}")
        print(f"Number of warmup rounds is {self.warmup_rounds}")

        for cur_round in range(self.rounds):
            print(f"\nRound number: {cur_round}")
            begin_round_time = time.time()
            self.cur_round = cur_round
            self.server.cur_round = cur_round
            if self.name_method == "FedDyn":
                self.server.dual_state_scale = self._get_dual_state_scale(cur_round)

            print("Evaluate started")
            server_val_loss = self.server.eval_global_model(need_save_loss=True)
            print(f"Server Val Loss: {server_val_loss}")

            # Algorithm 3, lines 2--5: resolve m, sample clients, update, aggregate.
            self.num_clients_subset = self.get_amount_clients()
            self.list_clients = self.server.select_clients_to_train(
                self.num_clients_subset
            )
            self.server.list_clients = self.list_clients
            self.list_clients.sort()
            if cur_round in self.find_optimal_rounds:
                self.server.full_round_clients = self.list_clients.copy()
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
        if self.name_method == "FedDyn":
            self.server.close_debug_files()

    def aggregate_clients(self, clients):
        """Apply Algorithm 5's aggregation rule to one candidate subset."""

        aggregated_weights = {
            key: value.detach().clone()
            for key, value in self.saved_weights.items()
        }

        coef = 1 / len(clients)

        for client in clients:
            for key, weights in self.client_gradients[client].items():
                update = weights.to(self.device)
                if aggregated_weights[key].is_floating_point():
                    aggregated_weights[key].add_(update, alpha=coef)
                else:
                    aggregated_weights[key] = (
                        aggregated_weights[key] + update * coef
                    )

        return aggregated_weights

    def fednova_aggregate_clients(self, clients):
        aggregated_weights = {
            key: value.detach().clone()
            for key, value in self.saved_weights.items()
        }
        ts = self.calculate_fednova_ts(clients)

        for client, weight in zip(clients, ts):
            for key, gradient in self.client_gradients[client].items():
                update = gradient.to(self.device)
                if aggregated_weights[key].is_floating_point():
                    aggregated_weights[key].add_(update, alpha=weight)
                else:
                    aggregated_weights[key] = (
                        aggregated_weights[key] + update * weight
                    )

        return aggregated_weights

    def get_mean_loss(self, amount_of_clients):
        """Run Algorithm 5's Monte Carlo objective estimate for one count."""
        losses = []
        before_sampling = {
            key: value.detach().clone()
            for key, value in self.global_model.state_dict().items()
        }
        for _ in range(self.num_samples):
            if self.use_stale_updates:
                # Legacy mode samples cached updates; the paper path uses fresh I.
                clients = self.select_clients_to_train(
                    amount_of_clients, server_sampling=True
                )
            else:
                clients = rand.sample(self.full_round_clients, amount_of_clients)
            print(f"Sample of clients: {sorted(clients)}")

            sample_weights = self.aggregate_clients(clients)
            self.global_model.load_state_dict(sample_weights)

            sample_loss = self.eval_global_model(need_save_loss=False)
            self.global_model.load_state_dict(before_sampling)

            print(f"Sample loss : {sample_loss}")
            losses.append(sample_loss)

        print(f"Sample losses = {losses}")

        mean_loss = sum(losses) / self.num_samples
        return mean_loss

    def ema_loss_history(self):
        if self.ema_use:
            # Get a trust_losses history until previous full communication round
            trust_losses_series = pd.Series(self.trust_losses[:-1])

            ema_trust_losses = list(
                trust_losses_series.ewm(span=self.window_sz_loss, adjust=False).mean()
            )
            # choose for comparsion last loss from ema history
            return ema_trust_losses[-1]

        return self.trust_losses[-2]

    def get_functional_value(self, amount_of_clients):
        mean_loss = self.get_mean_loss(amount_of_clients)
        print(f"Mean loss: {mean_loss}")

        func_value = self.first_positive_functional(mean_loss)
        print(f"Functional value = {func_value}")

        return func_value

    def choose_optimal(self, client_dict):
        optimal_amount_cl = None

        for k in client_dict.keys():
            if client_dict[k] > 0:
                optimal_amount_cl = k
                break
        if optimal_amount_cl is not None:
            # The optimization task has a solution.
            # Update with momentum
            if len(self.clients_history) == 0:
                optimal_amount_cl = optimal_amount_cl
            else:
                optimal_amount_cl = int(
                    np.ceil(
                        (1 - self.num_clients_momentum)
                        * self.clients_history[-1]
                        + self.num_clients_momentum * optimal_amount_cl
                    )
                )
        else:
            # The optimization task has no solution.
            # Take previous (or default=20) num of clients
            if len(self.clients_history) == 0:
                optimal_amount_cl = 20
            else:
                optimal_amount_cl = self.clients_history[-1]

        return optimal_amount_cl

    def find_optimal(self, borders_of_clients):
        """Run Algorithm 4's direct candidate-count search."""
        print(
            f"STARTING FIND OPTIMAL AMOUNT OF CLIENTS #{self.cnt_solve_optimal_task}:\n"
        )
        print(f"Trust loss history: {self.trust_losses}")
        print(f"Borders to solve problem: {borders_of_clients}\n")

        self.global_model.eval()
        saved_after_aggregate = copy.deepcopy(self.global_model.state_dict())

        client_dict = {
            k: 0.0
            for k in range(
                borders_of_clients[0], borders_of_clients[1], self.sample_step
            )
        }

        self.mean_prev_loss = self.ema_loss_history()
        print(f"Mean previously trust loss: {self.mean_prev_loss}")

        for k in range(borders_of_clients[0], borders_of_clients[1], self.sample_step):
            print(f"\nNow clients = {k}")
            # client_dict[k] = self.get_functional_value(k)
            delta_f = self.get_functional_value(k)
            client_dict[k] = delta_f
            print()
            if delta_f > 0:
                print(f"Find positive delta for {k} amount of clients")
                optimal_amount_cl = k
                break

        # update optimal_amount_cl
        optimal_amount_cl = self.choose_optimal(client_dict)

        print(f"OPTIMAL AMOUNT OF CLIENTS (optimal_amount_cl): {optimal_amount_cl}\n")

        self.global_model.load_state_dict(saved_after_aggregate)
        self.cnt_solve_optimal_task += 1
        return optimal_amount_cl

    def first_positive_functional(self, mean_loss):
        # The code stores -Phi, so positive values satisfy Algorithm 4's Phi < 0.
        d_loss = self.mean_prev_loss - mean_loss
        return d_loss

    def get_communication_content(self, rank):
        content = self.full_method_get_communication_content(rank)
        content["ISP_method_local_epoch"] = (
            self.local_epoch_on_wp
            if (self.cur_round < self.warmup_rounds)
            else self.cfg.federated_params.round_epochs
        )
        return content

    def client_create_pipe_commands(self):
        pipe_commands_map = self.full_method_create_pipe_commands()
        pipe_commands_map["ISP_method_local_epoch"] = self.set_local_epoch
        return pipe_commands_map

    def client_set_local_epoch(self, local_epoch):
        self.local_epoch = local_epoch

    def client_train_fn(self):
        # Algorithm 6, lines 3--10; BaseClient forms the transmitted update later.
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
