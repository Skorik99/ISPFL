import numpy as np


def make_candidate_grid(maximum: int, step: int) -> np.ndarray:
    grid = np.arange(1, maximum + 1, step, dtype=int)
    if grid[-1] != maximum:
        grid = np.append(grid, maximum)
    return grid


def posterior_query(
    grid: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    beta: float,
) -> tuple[int, str]:
    """Choose Algorithm 8's smallest feasible count or minimum upper bound."""
    upper_bound = mean + beta * standard_deviation
    feasible = np.flatnonzero(upper_bound < 0.0)
    if len(feasible):
        return int(grid[feasible[0]]), "first_confidence_feasible"
    return int(grid[np.argmin(upper_bound)]), "minimum_upper_bound"


def final_candidate(
    grid: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    beta: float,
    fallback: int,
) -> int:
    """Return Algorithm 8's final feasible count or the runtime fallback."""
    upper_bound = mean + beta * standard_deviation
    feasible = grid[upper_bound < 0.0]
    return int(feasible[0]) if len(feasible) else int(fallback)


def support_queries(
    grid: np.ndarray,
    event_index: int,
    query_count: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng([seed, event_index])
    strata = np.array_split(grid, min(query_count, len(grid)))
    return sorted(int(rng.choice(stratum)) for stratum in strata)
