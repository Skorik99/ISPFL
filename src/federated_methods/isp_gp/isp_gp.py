from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils.attack_utils import set_client_map_round

from ..fedavg.fedavg import FedAvg
from .acquisition import (
    final_candidate,
    make_candidate_grid,
    posterior_query,
    support_queries,
)
from .artifacts import ArtifactLogger
from .clients import ISPClient
from .context import prepare_context
from .full_client_evaluator import FullClientEvaluator
from .gp_model import GPSurrogate
from .proxy_cohort import proxy_label_distribution, select_proxy_cohort
from .servers import ISPServer


class ISPGP(FedAvg):
    """Paper ISP-GP with ProxyCohort evaluation on full client datasets."""

    def __init__(
        self,
        num_clients_subset,
        audit_interval,
        local_epochs_on_support,
        auxiliary_multiplier,
        sample_step,
        num_clients_momentum,
        support,
        posterior_path,
        online_posterior,
        proxy_cohort,
        evaluator,
        gp,
        logging,
    ):
        super().__init__(num_clients_subset=num_clients_subset)
        self.audit_interval = int(audit_interval)
        self.local_epochs_on_support = int(local_epochs_on_support)
        self.auxiliary_multiplier = float(auxiliary_multiplier)
        self.sample_step = int(sample_step)
        self.num_clients_momentum = float(num_clients_momentum)
        self.support_cfg = OmegaConf.to_container(support, resolve=True)
        self.posterior_cfg = OmegaConf.to_container(posterior_path, resolve=True)
        self.online_cfg = OmegaConf.to_container(online_posterior, resolve=True)
        self.proxy_cfg = OmegaConf.to_container(proxy_cohort, resolve=True)
        self.evaluator_cfg = OmegaConf.to_container(evaluator, resolve=True)
        self.gp_cfg = OmegaConf.to_container(gp, resolve=True)
        self.logging_cfg = OmegaConf.to_container(logging, resolve=True)

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)
        self.population_size = int(cfg.federated_params.amount_of_clients)
        self.optimal_amount_clients = int(self.num_clients_subset)
        self.burn_in_rounds = int(self.support_cfg["burn_in_rounds"])
        self.support_events = int(self.support_cfg["events"])
        self.support_end_round = self.burn_in_rounds + self.support_events
        self.support_pool = int(self.support_cfg["pool_size"])
        self.support_grid = [int(value) for value in self.support_cfg["grid"]]
        self.client_sizes = {
            int(client): int(size) for client, size in df.groupby("client").size().items()
        }
        self.client_sample_counts = np.asarray(
            [self.client_sizes[client] for client in range(self.population_size)],
            dtype=np.float64,
        )
        self.num_classes = int(cfg.training_params.num_classes)
        self.bias_parameter = str(self.proxy_cfg["bias_parameter"])
        self.proxy_updates = torch.zeros(
            self.population_size, self.num_classes, dtype=torch.float64
        )
        self.proxy_update_rounds = np.full(self.population_size, -1, dtype=int)
        self.evaluator = FullClientEvaluator(
            df,
            cfg,
            self.server.device,
            str(self.evaluator_cfg["tensor_cache"]),
        )
        self.gp_surrogate = GPSurrogate(
            training_steps=int(self.gp_cfg["training_steps"]),
            restarts=int(self.gp_cfg["restarts"]),
            learning_rate=float(self.gp_cfg["learning_rate"]),
            jitter=float(self.gp_cfg["jitter"]),
            seed=int(cfg.random_state),
        )
        self.observations = []
        self.gp_is_fitted = False
        self.audit_index = 0
        self.rng = np.random.default_rng(int(cfg.random_state))
        self.previous_global_update = None
        self.previous_loss_change = 0.0
        self.ema_loss = None
        self.ema_alpha = 2.0 / (float(self.online_cfg["ema_span"]) + 1.0)
        self._online_surrogate = None
        self._online_pending = None
        self._online_anchor_m = self.optimal_amount_clients
        self._online_auxiliary_size = self.support_pool
        self._online_log_action = math.log(self.optimal_amount_clients)
        output_dir = Path(str(cfg.single_run_dir)) / self.logging_cfg["output_subdir"]
        self.artifacts = ArtifactLogger(output_dir)
        self.gp_state_path = output_dir / "gp_state.pt"

    def _init_server(self, cfg):
        self.server = ISPServer(cfg)
        self.server.amount_classes = int(self.cfg.training_params.num_classes)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = ISPClient
        self.client_kwargs["client_cls"] = self.client_cls

    def _is_support_round(self, round_index):
        return self.burn_in_rounds <= round_index < self.support_end_round

    def _is_audit_round(self, round_index):
        if self._is_support_round(round_index):
            return True
        return round_index >= self.support_end_round and (
            round_index - self.support_end_round
        ) % self.audit_interval == 0

    def _auxiliary_size(self):
        if self._is_support_round(self.cur_round):
            return min(self.population_size, self.support_pool)
        return min(
            self.population_size,
            max(
                self.optimal_amount_clients,
                int(math.ceil(self.auxiliary_multiplier * self.optimal_amount_clients)),
            ),
        )

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        content["isp_task"] = {"local_epochs": self.current_local_epochs}
        return content

    def _selected_client_batches(self):
        batch_size = self.manager.batches.batch_size
        for start in range(0, len(self.list_clients), batch_size):
            yield self.list_clients[start : start + batch_size]

    def train_round(self):
        if not hasattr(self, "_worker_ranks"):
            self._worker_ranks = list(range(self.manager.batches.batch_size))
        for clients_batch in self._selected_client_batches():
            print(f"Current batch of clients is {clients_batch}", flush=True)
            for pipe_index, rank in enumerate(clients_batch):
                if self._worker_ranks[pipe_index] != rank:
                    self.server.pipes[pipe_index].send({"reinit": rank})
                    self._worker_ranks[pipe_index] = rank
                self.server.send_content_to_client(
                    pipe_index, self.get_communication_content(rank)
                )
            for pipe_index in range(len(clients_batch)):
                self.parse_communication_content(
                    self.server.rcv_content_from_client(pipe_index)
                )

    def _refresh_proxy_updates(self, round_index):
        # Algorithm 7 keeps each client's latest bias update; unseen clients are uniform.
        for client in self.list_clients:
            update = self.server.client_gradients[client]
            if self.bias_parameter not in update:
                raise KeyError(
                    f"ProxyCohort bias parameter '{self.bias_parameter}' is missing"
                )
            bias_update = update[self.bias_parameter].detach().cpu().reshape(-1)
            if len(bias_update) != self.num_classes:
                raise ValueError(
                    f"ProxyCohort expected {self.num_classes} bias values, "
                    f"received {len(bias_update)}"
                )
            self.proxy_updates[client] = bias_update.double()
            self.proxy_update_rounds[client] = round_index

    def _proxy_cohort(self, round_index):
        temperature = float(self.proxy_cfg["temperature"])
        distributions = np.stack(
            [
                proxy_label_distribution(update, temperature)
                for update in self.proxy_updates
            ]
        )
        cohort = select_proxy_cohort(
            distributions,
            self.client_sample_counts,
            float(self.proxy_cfg["coverage"]),
        )
        self.artifacts.append(
            "proxy_cohorts.csv",
            {
                "audit": self.audit_index,
                "round": round_index,
                "clients": json.dumps(cohort.clients),
                "size": len(cohort.clients),
                "coverage": float(self.proxy_cfg["coverage"]),
                "estimated_class_counts": json.dumps(cohort.covered_counts.tolist()),
                "uniform_prior_clients": int(
                    np.sum(self.proxy_update_rounds[cohort.clients] < 0)
                ),
            },
        )
        return cohort.clients

    def _sample_candidate_clients(self, auxiliary_clients, m):
        return sorted(
            self.rng.choice(auxiliary_clients, size=m, replace=False)
            .astype(int)
            .tolist()
        )

    def _evaluate_candidate(
        self,
        base_state,
        baseline_losses,
        auxiliary_clients,
        m,
        replicate_count,
        previous=None,
    ):
        """Estimate Algorithm 8's response at m from Monte Carlo subsets."""
        # Refinement retains earlier replicates and adds only the requested depth.
        responses = [] if previous is None else list(previous["_responses"])
        client_differences = (
            []
            if previous is None
            else [
                np.asarray(values, dtype=np.float64)
                for values in previous["_client_differences"]
            ]
        )
        subsets = [] if previous is None else list(previous["subsets"])
        for _ in range(replicate_count - len(responses)):
            clients = self._sample_candidate_clients(auxiliary_clients, m)
            candidate_state = self.server.aggregate_candidate(
                base_state, clients, self.client_sizes
            )
            candidate_losses = self.evaluator.evaluate_states(
                self.server.global_model, {"candidate": candidate_state}
            )["candidate"]
            response, differences = self.evaluator.response(
                candidate_losses, baseline_losses
            )
            responses.append(response)
            client_differences.append(differences)
            subsets.append(clients)
        responses = np.asarray(responses, dtype=np.float64)
        mean_differences = np.mean(client_differences, axis=0)
        candidate_variance = (
            float(responses.var(ddof=1) / len(responses))
            if len(responses) > 1
            else 0.0
        )
        cohort_variance = (
            float(mean_differences.var(ddof=1) / len(mean_differences))
            if len(mean_differences) > 1
            else 0.0
        )
        return {
            "m": int(m),
            "replicates": int(replicate_count),
            "response": float(responses.mean()),
            "noise_variance": max(
                candidate_variance + cohort_variance,
                float(self.evaluator_cfg["minimum_noise"]),
            ),
            "candidate_variance": candidate_variance,
            "cohort_variance": cohort_variance,
            "subsets": subsets,
            "_responses": responses.tolist(),
            "_client_differences": [values.tolist() for values in client_differences],
        }

    def _context(
        self, auxiliary_clients, baseline_loss, context_auxiliary_size=None
    ):
        return prepare_context(
            auxiliary_clients=auxiliary_clients,
            population_size=self.population_size,
            previous_m=self.optimal_amount_clients,
            updates=self.server.client_gradients,
            previous_update=self.previous_global_update,
            previous_loss_change=self.previous_loss_change,
            learning_rate=float(self.cfg.optimizer.lr),
            local_epochs=self.current_local_epochs,
            baseline_loss=baseline_loss,
            context_auxiliary_size=context_auxiliary_size,
        )

    def _append_observation(self, round_index, context, estimate):
        observation = {
            "round": int(round_index),
            "m": estimate["m"],
            "x": context.for_m(estimate["m"]),
            "y": estimate["response"],
            "v": estimate["noise_variance"],
            "replicates": estimate["replicates"],
            "candidate_variance": estimate["candidate_variance"],
        }
        self.observations.append(observation)
        self.artifacts.append(
            "gp_observations.csv",
            {
                "audit": self.audit_index,
                "round": round_index,
                "m": estimate["m"],
                "replicates": estimate["replicates"],
                "response": estimate["response"],
                "noise_variance": estimate["noise_variance"],
                "candidate_variance": estimate["candidate_variance"],
                "cohort_variance": estimate["cohort_variance"],
                "candidate_subsets": json.dumps(estimate["subsets"]),
            },
        )

    def _fit_gp_if_ready(self):
        minimum = int(self.gp_cfg["min_observations"])
        distinct_rounds = len({item["round"] for item in self.observations})
        if len(self.observations) < minimum or distinct_rounds < int(
            self.gp_cfg["min_distinct_rounds"]
        ):
            return False
        self.gp_surrogate.fit(
            np.stack([item["x"] for item in self.observations]),
            np.asarray([item["y"] for item in self.observations]),
            np.asarray([item["v"] for item in self.observations]),
        )
        self.gp_surrogate.save(self.gp_state_path)
        self.gp_is_fitted = True
        return True

    def _support_audit(
        self,
        round_index,
        base_state,
        baseline_losses,
        auxiliary_clients,
        context,
    ):
        grid = np.asarray(
            [m for m in self.support_grid if m <= len(auxiliary_clients)], dtype=int
        )
        queries = support_queries(
            grid,
            self.audit_index,
            int(self.support_cfg["queries_per_event"]),
            int(self.cfg.random_state),
        )
        for m in queries:
            estimate = self._evaluate_candidate(
                base_state,
                baseline_losses,
                auxiliary_clients,
                m,
                int(self.posterior_cfg["n_min"]),
            )
            self._append_observation(round_index, context, estimate)
        self._fit_gp_if_ready()

    def _conditioned_posterior(self, active, context, base_surrogate):
        """Condition historical D_GP on the current audit observations D^tau."""
        posterior = base_surrogate.clone()
        for m in sorted(active):
            estimate = active[m]
            posterior.condition(
                context.for_m(m)[None, :],
                np.asarray([estimate["response"]]),
                np.asarray([estimate["noise_variance"]]),
            )
        return posterior

    def _posterior_audit(
        self,
        round_index,
        base_state,
        baseline_losses,
        auxiliary_clients,
        context,
    ):
        """Run Algorithm 8's sequential posterior path and final GP refit."""
        grid = make_candidate_grid(len(auxiliary_clients), self.sample_step)
        contexts = context.for_grid(grid)
        decision_offset = float(self._decision_offset)
        base_surrogate = (
            self._online_surrogate
            if bool(self.online_cfg["use_as_audit_prior"])
            and self._online_surrogate is not None
            else self.gp_surrogate
        )
        active = {}
        step = 0
        while len(active) < int(self.posterior_cfg["q_max"]):
            posterior = self._conditioned_posterior(
                active, context, base_surrogate
            )
            prediction = posterior.predict(contexts)
            query, branch = posterior_query(
                grid,
                prediction.mean + decision_offset,
                prediction.standard_deviation,
                float(self.gp_cfg["decision_beta"]),
            )
            if query in active:
                # A repeated query refines its observation without growing |D^tau|.
                current_depth = int(active[query]["replicates"])
                if current_depth >= int(self.posterior_cfg["n_max"]):
                    break
                depth = min(
                    current_depth + int(self.posterior_cfg["n_step"]),
                    int(self.posterior_cfg["n_max"]),
                )
            else:
                depth = int(self.posterior_cfg["n_min"])
            estimate = self._evaluate_candidate(
                base_state,
                baseline_losses,
                auxiliary_clients,
                query,
                depth,
                active.get(query),
            )
            active[query] = estimate
            self.artifacts.append(
                "gp_posterior_path.csv",
                {
                    "audit": self.audit_index,
                    "round": round_index,
                    "step": step,
                    "query_m": query,
                    "replicates": depth,
                    "branch": branch,
                    "active_m": json.dumps(sorted(active)),
                },
            )
            step += 1

        for m in sorted(active):
            self._append_observation(round_index, context, active[m])
        self._fit_gp_if_ready()
        final_prediction = self.gp_surrogate.predict(contexts)
        raw_action = final_candidate(
            grid,
            final_prediction.mean + decision_offset,
            final_prediction.standard_deviation,
            float(self.gp_cfg["decision_beta"]),
            self.optimal_amount_clients,
        )
        previous_action = self.optimal_amount_clients
        action = int(
            math.floor(
                self.num_clients_momentum * raw_action
                + (1.0 - self.num_clients_momentum) * previous_action
            )
        )
        ratio_limit = self.posterior_cfg.get("action_ratio_limit")
        if ratio_limit is not None:
            lower = max(1, int(math.ceil(previous_action / float(ratio_limit))))
            upper = min(
                len(auxiliary_clients),
                int(math.floor(previous_action * float(ratio_limit))),
            )
            action = int(np.clip(action, lower, upper))
        self.optimal_amount_clients = max(1, min(action, len(auxiliary_clients)))
        nearest = min(active, key=lambda m: abs(m - previous_action))
        self.previous_loss_change = float(active[nearest]["response"])
        self.artifacts.append(
            "gp_decisions.csv",
            {
                "audit": self.audit_index,
                "round": round_index,
                "raw_action": raw_action,
                "previous_action": previous_action,
                "production_action": self.optimal_amount_clients,
            },
        )

    def _run_audit(self, round_index, base_state, auxiliary_clients):
        # Reuse one E_tau for the baseline and every candidate in this audit.
        evaluator_clients = self._proxy_cohort(round_index)
        self.evaluator.set_cohort(evaluator_clients)
        baseline_losses = self.evaluator.evaluate_states(
            self.server.global_model, {"baseline": base_state}
        )["baseline"]
        baseline_loss = float(self.evaluator.client_loss_means(baseline_losses).mean())
        self._decision_offset = baseline_loss - (
            baseline_loss if self.ema_loss is None else self.ema_loss
        )
        context = self._context(auxiliary_clients, baseline_loss)
        if self.gp_is_fitted and round_index >= self.support_end_round:
            self._posterior_audit(
                round_index,
                base_state,
                baseline_losses,
                auxiliary_clients,
                context,
            )
        else:
            self._support_audit(
                round_index,
                base_state,
                baseline_losses,
                auxiliary_clients,
                context,
            )
        self.audit_index += 1
        return baseline_loss, context

    def _online_candidate_variance(self, m):
        eligible = [
            observation
            for observation in self.observations
            if observation["replicates"] >= 2
        ]
        neighbors = sorted(
            eligible,
            key=lambda observation: abs(int(observation["m"]) - int(m)),
        )[: int(self.online_cfg["candidate_variance_neighbors"])]
        if not neighbors:
            return None
        single_subset_variances = [
            observation["candidate_variance"] * observation["replicates"]
            for observation in neighbors
        ]
        return float(np.median(single_subset_variances))

    def _online_panel(self, base_state):
        losses = self.evaluator.evaluate_states(
            self.server.global_model, {"online_baseline": base_state}
        )["online_baseline"]
        client_losses = self.evaluator.client_loss_means(losses)
        variance = (
            float(client_losses.var(ddof=1) / len(client_losses))
            if len(client_losses) > 1
            else 0.0
        )
        return {
            "mean": float(client_losses.mean()),
            "variance": variance,
            "client_losses": client_losses.tolist(),
        }

    def _reset_online_posterior(self, auxiliary_size):
        if not bool(self.online_cfg["enabled"]) or not self.gp_is_fitted:
            self._online_surrogate = None
            self._online_pending = None
            return
        self._online_surrogate = self.gp_surrogate.clone()
        self._online_anchor_m = int(self.optimal_amount_clients)
        self._online_auxiliary_size = int(auxiliary_size)
        self._online_log_action = math.log(self._online_anchor_m)
        self._online_pending = None

    def _online_observation(self, round_index, current_panel):
        pending = self._online_pending
        if pending is None:
            return None
        candidate_variance = self._online_candidate_variance(pending["m"])
        if candidate_variance is None:
            return None
        response = float(current_panel["mean"] - pending["panel"]["mean"])
        observation = {
            "source_round": pending["round"],
            "available_round": round_index,
            "m": pending["m"],
            "context": pending["context"],
            "response": response,
            "noise_variance": max(
                candidate_variance
                + current_panel["variance"]
                + pending["panel"]["variance"],
                float(self.evaluator_cfg["minimum_noise"]),
            ),
        }
        self.previous_loss_change = response
        self.artifacts.append(
            "gp_online_observations.csv",
            {
                **observation,
                "context": json.dumps(observation["context"].tolist()),
            },
        )
        return observation

    def _apply_online_action(self, round_index, context, panel_mean, observation):
        if self._online_surrogate is None:
            return
        if observation is not None:
            self._online_surrogate.condition(
                observation["context"][None, :],
                np.asarray([observation["response"]]),
                np.asarray([observation["noise_variance"]]),
            )
        grid = make_candidate_grid(self._online_auxiliary_size, self.sample_step)
        prediction = self._online_surrogate.predict(context.for_grid(grid))
        offset = panel_mean - (panel_mean if self.ema_loss is None else self.ema_loss)
        raw_action = final_candidate(
            grid,
            prediction.mean + offset,
            prediction.standard_deviation,
            float(self.gp_cfg["decision_beta"]),
            self.optimal_amount_clients,
        )
        ratio = float(self.online_cfg["action_ratio_limit"])
        lower = max(1, math.ceil(self._online_anchor_m / ratio))
        upper = min(
            self._online_auxiliary_size,
            math.floor(self._online_anchor_m * ratio),
        )
        clipped_action = int(np.clip(raw_action, lower, upper))
        momentum = float(self.online_cfg["momentum"])
        self._online_log_action = float(
            np.clip(
                (1.0 - momentum) * self._online_log_action
                + momentum * math.log(clipped_action),
                math.log(lower),
                math.log(upper),
            )
        )
        proposed_action = int(
            np.clip(round(math.exp(self._online_log_action)), lower, upper)
        )
        previous_action = self.optimal_amount_clients
        if bool(self.online_cfg["apply_action"]):
            self.optimal_amount_clients = proposed_action
        self.artifacts.append(
            "gp_online_actions.csv",
            {
                "round": round_index,
                "previous_action": previous_action,
                "raw_action": raw_action,
                "clipped_action": clipped_action,
                "production_action": self.optimal_amount_clients,
            },
        )

    def _run_online_posterior(
        self,
        round_index,
        base_state,
        actual_clients,
        audit_round,
        audit_context,
    ):
        panel = self._online_panel(base_state)
        observation = None if audit_round else self._online_observation(
            round_index, panel
        )
        if audit_round:
            self._reset_online_posterior(audit_context.auxiliary_size)
            decision_context = audit_context
        else:
            decision_context = self._context(
                actual_clients,
                panel["mean"],
                context_auxiliary_size=self._online_auxiliary_size,
            )
            if self.gp_is_fitted and bool(self.online_cfg["enabled"]):
                self._apply_online_action(
                    round_index, decision_context, panel["mean"], observation
                )
        self._online_pending = {
            "round": round_index,
            "m": len(actual_clients),
            "context": decision_context.for_m(len(actual_clients)),
            "panel": panel,
        }
        self.ema_loss = (
            panel["mean"]
            if self.ema_loss is None
            else self.ema_alpha * panel["mean"]
            + (1.0 - self.ema_alpha) * self.ema_loss
        )

    def begin_train(self):
        self.manager.create_clients(
            self.client_args, self.client_kwargs, self.client_attack_map
        )
        self.server.global_model = instantiate(self.cfg.models[0])
        for round_index in range(self.rounds):
            started_at = time.time()
            self.cur_round = round_index
            self.server.cur_round = round_index
            audit_round = self._is_audit_round(round_index)
            requested_clients = (
                self._auxiliary_size() if audit_round else self.optimal_amount_clients
            )
            self.current_local_epochs = (
                self.local_epochs_on_support
                if round_index < self.support_end_round
                else int(self.cfg.federated_params.round_epochs)
            )
            self.list_clients = sorted(
                self.server.select_clients_to_train(requested_clients)
            )
            self.server.list_clients = self.list_clients
            auxiliary_clients = self.list_clients.copy()
            audit_context = None
            base_state = copy.deepcopy(self.server.global_model.state_dict())
            self.client_map_round = set_client_map_round(
                self.client_attack_map,
                self.attack_rounds,
                self.attack_scheme,
                round_index,
            )
            # Audit rounds train the auxiliary pool once, then evaluate its subsets.
            self.train_round()
            self._refresh_proxy_updates(round_index)
            if audit_round:
                _, audit_context = self._run_audit(
                    round_index, base_state, auxiliary_clients
                )
                self.list_clients = self._sample_candidate_clients(
                    auxiliary_clients, self.optimal_amount_clients
                )
                self.server.list_clients = self.list_clients

            if self.evaluator.assignment is not None:
                self._run_online_posterior(
                    round_index,
                    base_state,
                    self.list_clients,
                    audit_round,
                    audit_context,
                )

            self.server.test_global_model()
            self.server.save_best_model(round_index)
            self.ts = self.calculate_ts()
            aggregated_state = self.aggregate()
            self.previous_global_update = {
                key: aggregated_state[key].detach().cpu()
                - base_state[key].detach().cpu()
                for key in base_state
            }
            self.server.global_model.load_state_dict(aggregated_state)
            print(f"Round time: {time.time() - started_at}", flush=True)
        print("Shutdown clients, federated learning end", flush=True)
        self.manager.stop_train()
