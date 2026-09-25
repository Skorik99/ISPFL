from ..fedavg.fedavg_client import FedAvgClient


class PowClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)

    def get_communication_content(self):
        content = super().get_communication_content()

        if self.need_train:
            # If client truly train, he need to send his validation loss
            self.client_val_loss, self.client_metrics = self.eval_fn()
            content["client_loss"] = self.client_val_loss

        return content
