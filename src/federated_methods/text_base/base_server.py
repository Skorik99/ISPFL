import math
import pandas as pd
import torch

from federated_methods.base.base_server import BaseServer
from utils.losses import get_loss


class TextBaseServer(BaseServer):
    def eval_fn(self):
        self.global_model.to(self.device)
        self.global_model.eval()
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.test_df,
        )
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for input, targets in self.iter_test_batches():
                inp = input[0]
                if (
                    self._cached_test_batches is None
                    or str(self._cache_device()) != str(self.device)
                ):
                    inp = inp.to(self.device)
                    targets = targets.to(self.device)
                outputs = self.global_model(inp)

                loss = self.get_loss_value(outputs, targets)
                total_loss += loss.detach().item()

                preds = outputs.argmax(dim=-1)
                mask = targets != self.criterion.ignore_index
                total_correct += (preds[mask] == targets[mask]).sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / len(self.test_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

        metrics = pd.DataFrame(
            {"shakespeare": [accuracy, perplexity]},
            index=["Accuracy", "Perplexity"],
        )
        return avg_loss, metrics

    def get_loss_value(self, outputs, targets):
        if outputs.dim() == 3:
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
        return self.criterion(outputs, targets)

    def test_global_model(self):
        print(f"\nServer Test Results:")
        self.test_loss, metrics = self.eval_fn()
        # Store as tuple to satisfy create_model_info interface (metrics, threshold)
        self.last_metrics = (metrics, None)
        print(metrics)
        print(f"Server Test Loss: {self.test_loss}")
