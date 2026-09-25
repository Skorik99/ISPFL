from ..fedavg.fedavg_client import FedAvgClient


class FedNovaClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.local_steps = 0

    def train(self):
        super().train()
        if self.need_train:
            local_epochs = getattr(
                self, "local_epoch", self.cfg.federated_params.round_epochs
            )
            self.local_steps = local_epochs * len(self.train_loader)

    def get_communication_content(self):
        content = super().get_communication_content()
        content["local_steps"] = self.local_steps if self.need_train else 0
        return content
