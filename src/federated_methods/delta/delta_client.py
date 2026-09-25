import time
import copy

from ..fedavg.fedavg_client import FedAvgClient


class DeltaClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)

    def get_communication_content(self):
        content = super().get_communication_content()
        content["batch_grads"] = None

        if not self.need_train:
            return content

        content["sigma"] = self.sigma
        return content

    def get_grad_by_batch(self):
        self.model.train()
        batch_count = 0
        mean_grads = {}
        squared_deviations = {}

        for batch in self.train_loader:
            _, (input, targets) = batch

            inp = input[0].to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inp)

            loss = self.get_loss_value(outputs, targets)

            loss.backward()

            batch_count += 1
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue

                grad = param.grad.detach()
                if batch_count == 1:
                    mean_grads[name] = grad.clone()
                    squared_deviations[name] = grad.new_zeros(grad.shape)
                    continue

                delta = grad - mean_grads[name]
                mean_grads[name].add_(delta, alpha=1.0 / batch_count)
                squared_deviations[name].addcmul_(
                    delta, grad - mean_grads[name]
                )

        total_squared_deviation = sum(
            tensor.sum() for tensor in squared_deviations.values()
        )
        return (total_squared_deviation / batch_count).sqrt().item()

    def train(self):
        if self.need_train:
            self.server_model_state = copy.deepcopy(self.model).state_dict()
            start = time.time()
            self.server_val_loss, self.server_metrics = self.eval_fn()
            self.train_fn()
            if self.print_metrics:
                self.client_val_loss, self.client_metrics = self.eval_fn()
            self.get_grad()
            self.result_time = time.time() - start

            # ------ DELTA ------ #
            self.sigma = self.get_grad_by_batch()
            # ------ DELTA ------ #
        else:
            self.server_val_loss, self.server_metrics = self.eval_fn()
