from ..fedavg.fedavg_client import FedAvgClient
from ..text_base.text_fedavg_client import TextFedAvgClient


class ISPClient(FedAvgClient):
    def __init__(self, *args, **kwargs):
        self.isp_task = {"local_epochs": 1}
        super().__init__(*args, **kwargs)

    def create_pipe_commands(self):
        commands = super().create_pipe_commands()
        commands["isp_task"] = self._set_isp_task
        return commands

    def _set_isp_task(self, task):
        self.isp_task = task
        self.need_train = True

    def train_fn(self):
        self.model.train()
        for _ in range(int(self.isp_task["local_epochs"])):
            for _, (inputs, targets) in self.train_loader:
                inputs = inputs[0].to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                loss = self.get_loss_value(self.model(inputs), targets)
                loss.backward()
                self.optimizer.step()


class ISPFedNovaClient(ISPClient):
    def train(self):
        super().train()
        self.local_steps = int(self.isp_task["local_epochs"]) * len(self.train_loader)

    def get_communication_content(self):
        content = super().get_communication_content()
        content["local_steps"] = self.local_steps
        return content


class ISPTextClient(TextFedAvgClient):
    def __init__(self, *args, **kwargs):
        self.isp_task = {"local_epochs": 1}
        super().__init__(*args, **kwargs)

    def create_pipe_commands(self):
        commands = super().create_pipe_commands()
        commands["isp_task"] = self._set_isp_task
        return commands

    def _set_isp_task(self, task):
        self.isp_task = task
        self.need_train = True

    def train_fn(self):
        self.model.train()
        for _ in range(int(self.isp_task["local_epochs"])):
            for _, (inputs, targets) in self.train_loader:
                inputs = inputs[0].to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                loss = self.get_loss_value(self.model(inputs), targets)
                loss.backward()
                self.optimizer.step()
