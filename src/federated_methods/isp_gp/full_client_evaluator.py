import copy

import numpy as np
import torch
import torch.nn.functional as functional

from utils.data_utils import get_dataset_loader


class FullClientEvaluator:
    """Evaluate models on complete client datasets held by the server simulator."""

    def __init__(self, dataframe, cfg, device, tensor_cache: str):
        self.dataframe = dataframe.reset_index(drop=True)
        self.cfg = cfg
        self.device = device
        self.tensor_cache = tensor_cache
        self.sequence_task = "shakespeare" in str(cfg.dataset.data_name)
        self.ignore_index = int(cfg.loss.config.ignore_index)
        self.label_smoothing = float(cfg.loss.config.label_smoothing)
        self.clients = []
        self.assignment = None
        self.client_indices = {}
        self.cached_batches = None

    def set_cohort(self, clients) -> None:
        self.clients = [int(client) for client in clients]
        self.assignment = self.dataframe[
            self.dataframe["client"].isin(self.clients)
        ].reset_index(drop=True)
        client_column = self.assignment["client"].to_numpy()
        self.client_indices = {
            client: np.flatnonzero(client_column == client) for client in self.clients
        }
        loader = get_dataset_loader(
            self.assignment, self.cfg, drop_last=False, mode="valid"
        )
        self.cached_batches = None
        if not self.sequence_task and self.tensor_cache != "disabled":
            cache_device = self.device if self.tensor_cache == "gpu" else "cpu"
            self.cached_batches = [
                (
                    indices.cpu(),
                    inputs[0].contiguous().to(cache_device),
                    targets.contiguous().to(cache_device),
                )
                for indices, (inputs, targets) in loader
            ]
        self.loader = loader

    def evaluate_states(self, model, states):
        original_state = copy.deepcopy(model.state_dict())
        was_training = model.training
        losses = {}
        try:
            model.to(self.device)
            model.eval()
            for state_id, state in states.items():
                model.load_state_dict(
                    {key: value.to(self.device) for key, value in state.items()}
                )
                losses[state_id] = self._per_sample_loss(model)
        finally:
            model.load_state_dict(original_state)
            model.train(was_training)
        return losses

    def _batches(self):
        if self.cached_batches is not None:
            yield from self.cached_batches
            return
        for indices, (inputs, targets) in self.loader:
            yield indices.cpu(), inputs[0], targets

    def _per_sample_loss(self, model):
        losses = torch.empty(len(self.assignment), dtype=torch.float32)
        with torch.no_grad():
            for indices, inputs, targets in self._batches():
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                logits = model(inputs)
                if self.sequence_task:
                    token_losses = functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        reduction="none",
                        ignore_index=self.ignore_index,
                        label_smoothing=self.label_smoothing,
                    ).reshape_as(targets)
                    mask = targets.ne(self.ignore_index)
                    batch_losses = (token_losses * mask).sum(dim=1) / mask.sum(
                        dim=1
                    ).clamp_min(1)
                else:
                    batch_losses = functional.cross_entropy(
                        logits,
                        targets,
                        reduction="none",
                        ignore_index=self.ignore_index,
                        label_smoothing=self.label_smoothing,
                    )
                losses[indices] = batch_losses.cpu()
        return losses

    def client_loss_means(self, losses) -> np.ndarray:
        return np.asarray(
            [float(losses[self.client_indices[client]].mean()) for client in self.clients]
        )

    def response(self, candidate_losses, baseline_losses) -> tuple[float, np.ndarray]:
        """Return the paired candidate-minus-baseline response for ExpectEstim."""
        differences = self.client_loss_means(candidate_losses) - self.client_loss_means(
            baseline_losses
        )
        return float(differences.mean()), differences
