from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PreparedContext:
    """Algorithm 7 context with the candidate-count feature filled per query."""

    base: np.ndarray
    auxiliary_size: int

    def for_m(self, m: int) -> np.ndarray:
        context = self.base.copy()
        context[0] = m / self.auxiliary_size
        return context

    def for_grid(self, grid: np.ndarray) -> np.ndarray:
        contexts = np.repeat(self.base[None, :], len(grid), axis=0)
        contexts[:, 0] = grid / self.auxiliary_size
        return contexts


def _tensor_dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left.detach().reshape(-1), right.detach().reshape(-1)))


def summarize_updates(
    updates: Sequence[Mapping[str, torch.Tensor]],
    clients: Sequence[int],
    previous_update: Mapping[str, torch.Tensor] | None,
) -> tuple[float, float, float]:
    """Compute Algorithm 7 update-norm statistics and previous-update alignment."""
    clients = list(clients)
    squared_norms = np.zeros(len(clients), dtype=np.float64)
    mean_squared_norm = 0.0
    previous_squared_norm = 0.0
    mean_previous_dot = 0.0

    for key, reference in updates[clients[0]].items():
        if not reference.is_floating_point():
            continue
        values = [updates[client][key].detach() for client in clients]
        for index, value in enumerate(values):
            squared_norms[index] += _tensor_dot(value, value)
        if previous_update is not None:
            mean_value = torch.zeros_like(reference)
            for value in values:
                mean_value.add_(value)
            mean_value.div_(len(values))
            previous_value = previous_update[key].detach()
            mean_squared_norm += _tensor_dot(mean_value, mean_value)
            previous_squared_norm += _tensor_dot(previous_value, previous_value)
            mean_previous_dot += _tensor_dot(mean_value, previous_value)

    norms = np.sqrt(np.maximum(squared_norms, 0.0))
    denominator = np.sqrt(mean_squared_norm * previous_squared_norm)
    cosine = 0.0 if denominator == 0.0 else mean_previous_dot / denominator
    return float(norms.mean()), float(norms.std()), float(cosine)


def prepare_context(
    *,
    auxiliary_clients: Sequence[int],
    population_size: int,
    previous_m: int,
    updates: Sequence[Mapping[str, torch.Tensor]],
    previous_update: Mapping[str, torch.Tensor] | None,
    previous_loss_change: float,
    learning_rate: float,
    local_epochs: int,
    baseline_loss: float,
    context_auxiliary_size: int | None = None,
) -> PreparedContext:
    """Construct the round-level context used by Algorithm 8."""
    mean_norm, norm_deviation, cosine = summarize_updates(
        updates, auxiliary_clients, previous_update
    )
    auxiliary_size = (
        len(auxiliary_clients)
        if context_auxiliary_size is None
        else int(context_auxiliary_size)
    )
    return PreparedContext(
        # m/b, previous m/M, b/M, norm mean/deviation, alignment,
        # previous loss change, learning rate, local epochs, baseline loss.
        base=np.asarray(
            [
                0.0,
                previous_m / population_size,
                auxiliary_size / population_size,
                mean_norm,
                norm_deviation,
                cosine,
                previous_loss_change,
                learning_rate,
                local_epochs,
                baseline_loss,
            ],
            dtype=np.float64,
        ),
        auxiliary_size=auxiliary_size,
    )
