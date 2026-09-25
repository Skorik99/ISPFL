from ..fedavg.fedavg_client import FedAvgClient


class OortClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.training_loss = None

    def train_fn(self):
        self.model.train()
        for _ in range(self.cfg.federated_params.round_epochs):

            for batch in self.train_loader:
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inp)

                loss = self.get_loss_value(outputs, targets)

                loss.backward()

                self.optimizer.step()

        ########### Oort additional ###########
        self.training_loss = loss.detach().item()
        #######################################

    def get_communication_content(self):
        content = super().get_communication_content()
        if self.need_train:
            content["training_loss"] = self.training_loss
        return content
