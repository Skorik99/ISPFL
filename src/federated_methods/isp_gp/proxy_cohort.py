from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ProxyCohort:
    clients: list[int]
    estimated_counts: np.ndarray
    covered_counts: np.ndarray


@dataclass(frozen=True)
class PopulationMatchingCohort:
    clients: list[int]
    mean_distribution: np.ndarray
    population_distribution: np.ndarray
    discrepancy: float


def proxy_label_distribution(
    bias_update: torch.Tensor | np.ndarray,
    temperature: float,
) -> np.ndarray:
    values = torch.as_tensor(bias_update, dtype=torch.float64).reshape(-1)
    return torch.softmax(values / temperature, dim=0).cpu().numpy()


def select_proxy_cohort(
    proxy_distributions: np.ndarray,
    client_sample_counts: np.ndarray,
    coverage: float,
) -> ProxyCohort:
    """Implement Algorithm 7's greedy proxy class-count coverage."""
    proxy_distributions = np.asarray(proxy_distributions, dtype=np.float64)
    client_sample_counts = np.asarray(client_sample_counts, dtype=np.float64)
    estimated_counts = client_sample_counts[:, None] * proxy_distributions
    deficits = np.full(proxy_distributions.shape[1], float(coverage))
    selected: list[int] = []

    while np.any(deficits > 1e-9):
        gains = np.minimum(estimated_counts, deficits).sum(axis=1)
        gains[selected] = -np.inf
        client = int(np.argmax(gains))
        if not np.isfinite(gains[client]) or gains[client] <= 0.0:
            raise ValueError("Proxy class coverage is infeasible")
        selected.append(client)
        deficits = np.maximum(0.0, deficits - estimated_counts[client])

    return ProxyCohort(
        clients=selected,
        estimated_counts=estimated_counts,
        covered_counts=estimated_counts[selected].sum(axis=0),
    )


def select_population_matching_cohort(
    proxy_distributions: np.ndarray,
) -> PopulationMatchingCohort:
    """Shakespeare policy: match the population proxy mean within L1 0.05."""
    proxy_distributions = np.asarray(proxy_distributions, dtype=np.float64)
    population_distribution = proxy_distributions.mean(axis=0)
    selected: list[int] = []

    while True:
        candidates = [
            client
            for client in range(len(proxy_distributions))
            if client not in selected
        ]
        discrepancies = [
            np.linalg.norm(
                proxy_distributions[selected + [client]].mean(axis=0)
                - population_distribution,
                ord=1,
            )
            for client in candidates
        ]
        selected.append(candidates[int(np.argmin(discrepancies))])
        mean_distribution = proxy_distributions[selected].mean(axis=0)
        discrepancy = float(
            np.linalg.norm(mean_distribution - population_distribution, ord=1)
        )
        if discrepancy <= 0.05:
            break

    return PopulationMatchingCohort(
        clients=selected,
        mean_distribution=mean_distribution,
        population_distribution=population_distribution,
        discrepancy=discrepancy,
    )
