import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from federated_methods.base.base_client import BaseClient
from utils.data_utils import get_dataset_loader


class TextBaseClient(BaseClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.ignore_index = getattr(self.criterion, "ignore_index", -100)

    def _init_loaders(self):
        client_df = self.df[self.df["client"] == self.rank]
        self.init_pos_weight = False

        if len(client_df) < 2 or self.train_val_prop == 0:
            self.train_df = client_df.reset_index(drop=True)
            self.valid_df = client_df.reset_index(drop=True)
        else:
            self.train_df, self.valid_df = train_test_split(
                client_df,
                test_size=self.train_val_prop,
                random_state=self.cfg.random_state,
                shuffle=True,
            )

        self.train_loader = get_dataset_loader(self.train_df, self.cfg, drop_last=False)
        self.valid_loader = get_dataset_loader(
            self.valid_df, self.cfg, drop_last=False, mode="valid"
        )

    def get_loss_value(self, outputs, targets):
        if outputs.dim() == 3:
            outputs = outputs.reshape(-1, outputs.size(-1))
        if targets.dim() > 1:
            targets = targets.reshape(-1)
        return self.criterion(outputs, targets)

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

    def eval_fn(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for _, batch in enumerate(self.valid_loader):
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inp)
                loss = self.get_loss_value(outputs, targets)
                total_loss += loss.detach().item()

                preds = outputs.argmax(dim=-1)
                mask = targets != self.ignore_index
                total_correct += (preds[mask] == targets[mask]).sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / len(self.valid_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        client_metrics = pd.DataFrame({"shakespeare": [accuracy]}, index=["Accuracy"])
        return avg_loss, client_metrics
