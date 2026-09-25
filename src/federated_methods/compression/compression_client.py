from ..fedavg.fedavg_client import FedAvgClient
import torch


class CompressionClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]  # `cfg` and `df`
        super().__init__(*base_client_args, **client_kwargs)
        self.client_args = client_args

        self.need_train = False
        self.compression_type = self.client_args[2]
        self.compression_k_percent = self.client_args[3]

    def get_grad(self):
        self.model.eval()
        for key, _ in self.model.state_dict().items():
            self.grad[key] = (
                self.model.state_dict()[key] - self.server_model_state[key]
            ).to("cpu")

            if self.model.state_dict()[key].dtype == torch.float32:
                self.grad[key] = self.compress(self.grad[key])

    def compress(self, weight):
        copy_weight = torch.clone(weight).detach()
        if self.compression_type == "topk":
            return self.topk_compress(copy_weight)
        elif self.compression_type == "randk":
            return self.randk_compress(copy_weight)
        else:
            raise NotImplementedError(
                f"Now we support only `randk`, `topk`\nYou provide: {self.compression_type}"
            )

    def topk_compress(self, weight):
        flat_weight = weight.flatten()
        k = min(
            int(self.compression_k_percent * flat_weight.numel() / 100),
            flat_weight.numel(),
        )

        topk_values, topk_indices = torch.topk(flat_weight.abs(), k)

        mask = torch.zeros_like(flat_weight, dtype=torch.float32)
        mask[topk_indices] = 1.0

        compressed_weight = flat_weight * mask
        return compressed_weight.view_as(weight)

    def randk_compress(self, weight):
        flat_weight = weight.flatten()
        k = min(
            int(self.compression_k_percent * flat_weight.numel() / 100),
            flat_weight.numel(),
        )

        randk_indices = torch.randperm(flat_weight.numel())[:k]

        mask = torch.zeros_like(flat_weight, dtype=torch.float32)
        mask[randk_indices] = 1.0

        compressed_weight = flat_weight * mask
        return compressed_weight.view_as(weight)
