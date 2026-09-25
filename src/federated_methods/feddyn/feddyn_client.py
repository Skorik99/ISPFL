from collections import OrderedDict

import torch
from torch.nn.utils import parameters_to_vector

from ..fedavg.fedavg_client import FedAvgClient


class FedDynClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.alpha = float(self.client_args[2])
        self.local_dual_state = self._build_zero_dual_state()
        self.local_dual_vector = self._state_to_vector(self.local_dual_state)
        self.server_model_vector = None
        self.alpha_adaptive = self._get_alpha_adaptive()
        self.debug_dual_state = bool(self.cfg.federated_method.debug_dual_state)
        self.dual_state_debug = None
        self.dual_state_scale = 1.0

    def _trainable_parameters(self):
        return [
            param
            for param in self.model.parameters()
            if param.requires_grad
        ]

    def _state_to_vector(self, state):
        return torch.cat(
            [
                state[name].detach().reshape(-1).to(self.device)
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ]
        )

    def _build_zero_dual_state(self):
        return OrderedDict(
            (
                name,
                torch.zeros_like(param.detach(), device=self.device),
            )
            for name, param in self.model.named_parameters()
            if param.requires_grad
        )

    def _get_alpha_adaptive(self):
        client_data_size = len(self.df[self.df["client"] == self.rank])
        total_data_size = max(1, len(self.df))
        amount_of_clients = self.cfg.federated_params.amount_of_clients
        client_weight = (
            client_data_size / total_data_size * amount_of_clients
            if client_data_size > 0
            else 1.0
        )
        return self.alpha / max(client_weight, 1e-12)

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["local_dual_state"] = self.set_local_dual_state
        pipe_commands_map["dual_state_scale"] = self.set_dual_state_scale
        return pipe_commands_map

    def set_dual_state_scale(self, dual_state_scale):
        self.dual_state_scale = float(dual_state_scale)

    def set_local_dual_state(self, local_dual_state):
        if local_dual_state is None:
            self.local_dual_state = self._build_zero_dual_state()
            self.local_dual_vector = self._state_to_vector(self.local_dual_state)
            return

        self.local_dual_state = OrderedDict(
            (
                name,
                tensor.to(self.device),
            )
            for name, tensor in local_dual_state.items()
        )
        self.local_dual_vector = self._state_to_vector(self.local_dual_state)

    def _get_server_model_vector(self):
        if self.server_model_vector is None:
            self.server_model_vector = self._state_to_vector(self.server_model_state)
        return self.server_model_vector

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)

        parameter_vector = parameters_to_vector(self._trainable_parameters())
        server_model_vector = self._get_server_model_vector()
        linear_term = torch.dot(
            parameter_vector,
            self.dual_state_scale * self.local_dual_vector - server_model_vector,
        )
        l2_term = torch.dot(parameter_vector, parameter_vector)

        return loss + self.alpha_adaptive * linear_term + 0.5 * self.alpha_adaptive * l2_term

    def train(self):
        # The server model changes every round, so cache its vector per local train call.
        self.server_model_vector = None

        if not self.need_train:
            super().train()
            return

        if self.debug_dual_state:
            server_model_vector = parameters_to_vector(
                self._trainable_parameters()
            ).detach()
            self.server_model_vector = server_model_vector
            dual_correction = (
                self.alpha_adaptive
                * self.dual_state_scale
                * self.local_dual_vector
            )
            self.dual_state_debug = {
                "client_id": self.rank,
                "client_size": len(self.df[self.df["client"] == self.rank]),
                "alpha_adaptive": self.alpha_adaptive,
                "dual_state_scale": self.dual_state_scale,
                "dual_state_norm": torch.linalg.vector_norm(
                    self.local_dual_vector
                ).item(),
                "dual_correction_norm": torch.linalg.vector_norm(
                    dual_correction
                ).item(),
            }

        super().train()

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.local_dual_state[name].add_(param.detach())
            self.local_dual_state[name].add_(
                self.server_model_state[name],
                alpha=-1.0,
            )

        if self.debug_dual_state:
            update_vector = parameters_to_vector(self._trainable_parameters())
            update_vector = update_vector - self._get_server_model_vector()
            self.dual_state_debug["local_update_norm"] = torch.linalg.vector_norm(
                update_vector
            ).item()
            self.dual_state_debug["r_i"] = self.dual_state_debug[
                "dual_correction_norm"
            ] / max(self.dual_state_debug["local_update_norm"], 1e-12)

    def get_communication_content(self):
        content = super().get_communication_content()
        content["local_dual_state"] = None

        if not self.need_train:
            return content

        if self.debug_dual_state:
            content["dual_state_debug"] = self.dual_state_debug

        content["local_dual_state"] = {
            name: tensor.detach().cpu() for name, tensor in self.local_dual_state.items()
        }
        return content
