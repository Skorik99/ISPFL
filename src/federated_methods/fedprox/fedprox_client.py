import torch

from ..fedavg.fedavg_client import FedAvgClient


class FedProxClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.mu = float(self.client_args[2])

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)

        proximal_term = torch.zeros((), device=self.device)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            proximal_term = proximal_term + torch.sum(
                (param - self.server_model_state[name]) ** 2
            )

        return loss + 0.5 * self.mu * proximal_term
