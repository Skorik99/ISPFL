from typing import Dict, List, Optional, Sequence

import numpy as np

UNIFORM_SELECTOR = "uniform"
FEDCBS_SELECTOR = "fedcbs"
DELTA_SELECTOR = "delta"
POW_SELECTOR = "pow"
FEDCOR_SELECTOR = "fedcor"
OORT_SELECTOR = "oort"


def _as_pool(
    client_pool: Optional[Sequence[int]], amount_of_clients: int
) -> np.ndarray:
    if client_pool is None:
        return np.arange(amount_of_clients, dtype=np.int64)
    pool = np.asarray(client_pool, dtype=np.int64)
    if pool.ndim != 1:
        raise ValueError("client_pool must be one-dimensional")
    if len(pool) == 0:
        raise ValueError("client_pool must be non-empty")
    return pool


def _normalize_probabilities(values: np.ndarray) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    total = probs.sum()
    if (not np.isfinite(total)) or total <= 0:
        return np.ones_like(probs, dtype=np.float64) / float(len(probs))
    probs = probs / total
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    total = probs.sum()
    if total <= 0:
        return np.ones_like(probs, dtype=np.float64) / float(len(probs))
    return probs / total


def _softmax_np(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    logits = logits - np.max(logits)
    exp_values = np.exp(logits)
    return _normalize_probabilities(exp_values)


def _resolve_subset_size(num_clients_subset: int, client_pool: np.ndarray) -> int:
    return int(min(num_clients_subset, len(client_pool)))


def uniform_select(
    *,
    num_clients_subset: int,
    amount_of_clients: int,
    client_pool: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    if rng is None:
        rng = np.random.default_rng()

    pool = _as_pool(client_pool, amount_of_clients)
    subset_size = _resolve_subset_size(num_clients_subset, pool)
    if subset_size >= len(pool):
        return pool.tolist()

    return rng.choice(pool, size=subset_size, replace=False).astype(np.int64).tolist()


def fedcbs_select(
    *,
    num_clients_subset: int,
    qcid_mtr: np.ndarray,
    selection_counter: np.ndarray,
    client_data_count: np.ndarray,
    amount_classes: int,
    cur_round: int,
    lambda_: float,
    amount_of_clients: int,
    client_pool: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    if rng is None:
        rng = np.random.default_rng()

    pool = _as_pool(client_pool, amount_of_clients)
    subset_size = _resolve_subset_size(num_clients_subset, pool)
    if subset_size >= len(pool):
        return pool.tolist()

    qcid_mtr = np.asarray(qcid_mtr, dtype=np.float64)
    selection_counter = np.asarray(selection_counter, dtype=np.float64)
    client_data_count = np.asarray(client_data_count, dtype=np.float64)

    qcid_diag = np.diag(qcid_mtr)
    selected: List[int] = []
    remaining = pool.copy()
    sum_qcid_ss = 0.0
    sum_counts_s = 0.0
    eps = 1e-12
    inv_classes = 1.0 / float(amount_classes)
    betas = np.arange(1, subset_size + 1, dtype=np.float64)

    for m in range(subset_size):
        if m == 0:
            denom = np.maximum(client_data_count[remaining] ** 2, eps)
            qcid_single = qcid_diag[remaining] / denom - inv_classes
            qcid_single = np.maximum(qcid_single, eps)
            explore = lambda_ * np.sqrt(
                3.0
                * np.log(max(1, int(cur_round)) + 1.0)
                / (2.0 * np.maximum(selection_counter[remaining], 1.0))
            )
            probs = 1.0 / (qcid_single ** betas[0]) + explore
        else:
            cross = qcid_mtr[
                np.ix_(remaining, np.asarray(selected, dtype=np.int64))
            ].sum(axis=1)
            sum_qcid_new = sum_qcid_ss + 2.0 * cross + qcid_diag[remaining]
            sum_counts_new = sum_counts_s + client_data_count[remaining]
            denom = np.maximum(sum_counts_new**2, eps)
            qcid_with_c = sum_qcid_new / denom - inv_classes
            qcid_with_c = np.maximum(qcid_with_c, eps)
            qcid_s = (
                sum_qcid_ss / max(sum_counts_s**2, eps) - inv_classes
                if sum_counts_s > 0
                else eps
            )
            qcid_s = max(qcid_s, eps)

            if m == 1:
                explore = lambda_ * np.sqrt(
                    3.0
                    * np.log(max(1, int(cur_round)) + 1.0)
                    / (2.0 * np.maximum(selection_counter[remaining], 1.0))
                )
                probs = (1.0 / (qcid_with_c ** betas[1])) / (
                    1.0 / (qcid_s ** betas[0]) + explore
                )
            else:
                probs = ((qcid_s / qcid_with_c) ** betas[m - 2]) / qcid_with_c

        probs = np.maximum(np.nan_to_num(probs, nan=eps, posinf=eps, neginf=eps), eps)
        probs = _normalize_probabilities(probs)

        chosen_idx = int(rng.choice(len(remaining), p=probs))
        chosen = int(remaining[chosen_idx])

        if selected:
            cross_chosen = qcid_mtr[chosen, np.asarray(selected, dtype=np.int64)].sum()
        else:
            cross_chosen = 0.0

        sum_qcid_ss += 2.0 * cross_chosen + qcid_diag[chosen]
        sum_counts_s += client_data_count[chosen]
        selected.append(chosen)
        remaining = np.delete(remaining, chosen_idx)

    return selected


def delta_select(
    *,
    num_clients_subset: int,
    client_probs: Optional[np.ndarray],
    amount_of_clients: int,
    client_pool: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    if rng is None:
        rng = np.random.default_rng()

    pool = _as_pool(client_pool, amount_of_clients)
    subset_size = _resolve_subset_size(num_clients_subset, pool)
    if subset_size >= len(pool):
        return pool.tolist()

    if client_probs is None:
        probs = np.ones(len(pool), dtype=np.float64) / float(len(pool))
    else:
        full_probs = np.asarray(client_probs, dtype=np.float64)
        probs = _normalize_probabilities(full_probs[pool])

    return (
        rng.choice(pool, size=subset_size, replace=False, p=probs)
        .astype(np.int64)
        .tolist()
    )


def pow_select(
    *,
    num_clients_subset: int,
    candidate_set_size: int,
    clients_probs: Optional[np.ndarray],
    clients_losses: Optional[np.ndarray],
    amount_of_clients: int,
    client_pool: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
    topk_tau: float = 0.0,
) -> List[int]:
    if rng is None:
        rng = np.random.default_rng()

    pool = _as_pool(client_pool, amount_of_clients)
    subset_size = _resolve_subset_size(num_clients_subset, pool)
    if subset_size >= len(pool):
        return pool.tolist()

    candidate_set_size = int(min(max(1, candidate_set_size), len(pool)))
    if clients_probs is None:
        probs = np.ones(len(pool), dtype=np.float64) / float(len(pool))
    else:
        full_probs = np.asarray(clients_probs, dtype=np.float64)
        probs = _normalize_probabilities(full_probs[pool])

    if clients_losses is None:
        full_losses = np.zeros(amount_of_clients, dtype=np.float64)
    else:
        full_losses = np.asarray(clients_losses, dtype=np.float64)

    candidate_pool = rng.choice(
        pool, size=candidate_set_size, replace=False, p=probs
    ).astype(np.int64)
    candidate_losses = full_losses[candidate_pool]

    if topk_tau is None or topk_tau <= 0 or not np.isfinite(topk_tau):
        ordered = candidate_pool[np.argsort(candidate_losses)[::-1]]
        return ordered[:subset_size].tolist()

    logits = candidate_losses / float(topk_tau)
    gumbel = -np.log(-np.log(rng.random(size=len(candidate_pool)) + 1e-30) + 1e-30)
    keys = logits + gumbel
    top_idx = np.argpartition(-keys, kth=subset_size - 1)[:subset_size]
    top_idx = top_idx[np.argsort(keys[top_idx])[::-1]]
    return candidate_pool[top_idx].tolist()


def fedcor_select(
    *,
    num_clients_subset: int,
    amount_of_clients: int,
    cur_round: int,
    warmup: int,
    covariance: Optional[np.ndarray],
    discount: Optional[np.ndarray],
    ts: Optional[np.ndarray],
    client_pool: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
    tau: float = 0.0,
) -> List[int]:
    if rng is None:
        rng = np.random.default_rng()

    pool = _as_pool(client_pool, amount_of_clients)
    subset_size = _resolve_subset_size(num_clients_subset, pool)
    if subset_size >= len(pool):
        return pool.tolist()

    if cur_round <= warmup or covariance is None:
        return (
            rng.choice(pool, size=subset_size, replace=False).astype(np.int64).tolist()
        )

    sigma = np.asarray(covariance, dtype=np.float64)[np.ix_(pool, pool)].copy()
    if discount is None:
        discount_local = np.ones(len(pool), dtype=np.float64)
    else:
        discount_local = np.asarray(discount, dtype=np.float64)[pool].copy()

    if ts is None:
        weights_local = None
    else:
        weights_local = np.asarray(ts, dtype=np.float64)[pool].copy()

    remain_local = np.arange(len(pool), dtype=np.int64)
    selected: List[int] = []

    for _ in range(subset_size):
        diag = np.clip(np.diag(sigma), a_min=1e-12, a_max=None)
        scaled_discount = discount_local / np.sqrt(diag)
        if weights_local is None:
            scores = sigma.sum(axis=0) * scaled_discount
        else:
            scores = (weights_local[:, None] * sigma).sum(axis=0) * scaled_discount

        if tau is None or tau <= 0 or not np.isfinite(tau):
            chosen_local = int(np.argmax(scores))
        else:
            chosen_local = int(
                rng.choice(len(remain_local), p=_softmax_np(scores / tau))
            )

        chosen_global = int(pool[remain_local[chosen_local]])
        selected.append(chosen_global)

        denom = float(sigma[chosen_local, chosen_local])
        if denom > 1e-12 and sigma.shape[0] > 1:
            column = sigma[:, chosen_local : chosen_local + 1]
            row = sigma[chosen_local : chosen_local + 1, :]
            sigma = sigma - (column @ row) / denom

        sigma = np.delete(np.delete(sigma, chosen_local, axis=0), chosen_local, axis=1)
        remain_local = np.delete(remain_local, chosen_local)
        discount_local = np.delete(discount_local, chosen_local)
        if weights_local is not None:
            weights_local = np.delete(weights_local, chosen_local)

        if len(remain_local) == 0:
            break

    return selected


def oort_select(
    *,
    num_clients_subset: int,
    training_cl_losses: np.ndarray,
    last_participation_round: np.ndarray,
    round_times: np.ndarray,
    cur_round: int,
    alpha_stale: float,
    loss_prior: float,
    c: float,
    preferred_time: float,
    penalty_degree: float,
    amount_of_clients: int,
    client_pool: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    if rng is None:
        rng = np.random.default_rng()

    pool = _as_pool(client_pool, amount_of_clients)
    subset_size = _resolve_subset_size(num_clients_subset, pool)
    if subset_size >= len(pool):
        return pool.tolist()

    losses = np.asarray(training_cl_losses, dtype=np.float64)[pool]
    last_rounds = np.asarray(last_participation_round, dtype=np.float64)[pool]
    times = np.asarray(round_times, dtype=np.float64)[pool]

    utilities = np.array(
        [loss if np.isfinite(loss) else loss_prior for loss in losses],
        dtype=np.float64,
    )

    sys_coef = []
    for value in times:
        if np.isfinite(value) and value > 0 and value < preferred_time:
            sys_coef.append((preferred_time / value) ** penalty_degree)
        else:
            sys_coef.append(1.0)
    utilities = utilities * np.asarray(sys_coef, dtype=np.float64)

    t_cur = max(1, int(cur_round) + 1)
    stale = np.maximum(last_rounds, 1.0)
    incentive = float(alpha_stale) * np.sqrt(np.log(t_cur) / stale)
    scores = utilities + incentive
    scores = np.minimum(scores, np.percentile(scores, 95))

    sort_idx = np.argsort(-scores)
    q_idx = min(max(subset_size - 1, 0), len(sort_idx) - 1)
    cutoff = c * scores[sort_idx[q_idx]]

    high_utility_mask = scores >= cutoff
    candidate_local = np.where(high_utility_mask)[0]
    if len(candidate_local) < subset_size:
        candidate_local = sort_idx[: max(subset_size, len(candidate_local))]

    probs = _normalize_probabilities(scores[candidate_local])
    chosen_local = rng.choice(
        candidate_local, size=subset_size, replace=False, p=probs
    ).astype(np.int64)
    return pool[chosen_local].tolist()


def resolve_selector_name(method) -> str:
    base_method = getattr(method, "base_method", None)
    if isinstance(base_method, str):
        selector_name = base_method.lower()
        if selector_name in {
            "fedavg",
            "fedprox",
            "fednova",
            "feddyn",
            "compression",
            "text_fedavg",
        }:
            return UNIFORM_SELECTOR
        return selector_name

    server = getattr(method, "server", None)
    if server is None:
        return UNIFORM_SELECTOR

    if hasattr(server, "qcid_mtr"):
        return FEDCBS_SELECTOR
    if hasattr(server, "gpr"):
        return FEDCOR_SELECTOR
    if hasattr(server, "training_cl_losses"):
        return OORT_SELECTOR
    if hasattr(server, "candidate_set_size"):
        return POW_SELECTOR
    if hasattr(server, "client_probs"):
        return DELTA_SELECTOR
    return UNIFORM_SELECTOR


def build_selector_context(method) -> Dict[str, object]:
    selector_name = resolve_selector_name(method)
    server = getattr(method, "server", None)
    amount_of_clients = int(getattr(method, "amount_of_clients", 0))
    cur_round = int(getattr(server, "cur_round", getattr(method, "cur_round", 0)))

    context: Dict[str, object] = {
        "selector_name": selector_name,
        "amount_of_clients": amount_of_clients,
        "cur_round": cur_round,
    }

    if selector_name == FEDCBS_SELECTOR:
        if isinstance(server.client_data_count, dict):
            client_data_count = np.array(
                [server.client_data_count[idx] for idx in range(amount_of_clients)],
                dtype=np.float64,
            )
        else:
            client_data_count = np.asarray(server.client_data_count, dtype=np.float64)

        context.update(
            {
                "qcid_mtr": np.asarray(server.qcid_mtr, dtype=np.float64),
                "selection_counter": np.asarray(
                    server.selection_counter, dtype=np.float64
                ),
                "client_data_count": client_data_count,
                "amount_classes": int(server.amount_classes),
                "lambda_": float(server.lambda_),
            }
        )
        return context

    if selector_name == DELTA_SELECTOR:
        context["client_probs"] = np.asarray(server.client_probs, dtype=np.float64)
        return context

    if selector_name == POW_SELECTOR:
        context.update(
            {
                "candidate_set_size": int(server.candidate_set_size),
                "clients_probs": np.asarray(server.clients_probs, dtype=np.float64),
                "clients_losses": np.asarray(server.clients_losses, dtype=np.float64),
            }
        )
        return context

    if selector_name == FEDCOR_SELECTOR:
        warmup = int(getattr(server, "warmup", 0))
        covariance = None
        discount = None
        if getattr(server, "gpr", None) is not None and cur_round > warmup:
            covariance = (
                server.gpr.Covariance().detach().cpu().numpy().astype(np.float64)
            )
            discount_tensor = getattr(server.gpr, "discount", None)
            if discount_tensor is not None:
                discount = discount_tensor.detach().cpu().numpy().astype(np.float64)

        ts = getattr(server, "ts", None)
        context.update(
            {
                "warmup": warmup,
                "covariance": covariance,
                "discount": discount,
                "ts": None if ts is None else np.asarray(ts, dtype=np.float64),
            }
        )
        return context

    if selector_name == OORT_SELECTOR:
        training_losses = np.array(
            [
                np.nan if loss is None else float(loss)
                for loss in server.training_cl_losses
            ],
            dtype=np.float64,
        )
        last_rounds = np.asarray(server.last_participation_round, dtype=np.float64)
        round_times = np.array(
            [np.nan if value is None else float(value) for value in server.round_times],
            dtype=np.float64,
        )
        context.update(
            {
                "training_cl_losses": training_losses,
                "last_participation_round": last_rounds,
                "round_times": round_times,
                "alpha_stale": float(server.alpha_stale),
                "loss_prior": float(server.loss_prior),
                "c": float(server.c),
                "preferred_time": float(server.preffered_time),
                "penalty_degree": float(server.penalty_degree),
            }
        )
        return context

    return context


def sample_clients(
    selector_context: Dict[str, object],
    num_clients_subset: int,
    *,
    client_pool: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
    fedcor_tau: float = 0.0,
    pow_topk_tau: float = 0.0,
) -> List[int]:
    selector_name = str(selector_context["selector_name"]).lower()
    amount_of_clients = int(selector_context["amount_of_clients"])

    if selector_name in {
        UNIFORM_SELECTOR,
        "compression",
        "fedavg",
        "fedprox",
        "fednova",
        "feddyn",
    }:
        return uniform_select(
            num_clients_subset=num_clients_subset,
            amount_of_clients=amount_of_clients,
            client_pool=client_pool,
            rng=rng,
        )

    if selector_name == FEDCBS_SELECTOR:
        return fedcbs_select(
            num_clients_subset=num_clients_subset,
            qcid_mtr=selector_context["qcid_mtr"],
            selection_counter=selector_context["selection_counter"],
            client_data_count=selector_context["client_data_count"],
            amount_classes=int(selector_context["amount_classes"]),
            cur_round=int(selector_context["cur_round"]),
            lambda_=float(selector_context["lambda_"]),
            amount_of_clients=amount_of_clients,
            client_pool=client_pool,
            rng=rng,
        )

    if selector_name == DELTA_SELECTOR:
        return delta_select(
            num_clients_subset=num_clients_subset,
            client_probs=selector_context.get("client_probs"),
            amount_of_clients=amount_of_clients,
            client_pool=client_pool,
            rng=rng,
        )

    if selector_name == POW_SELECTOR:
        return pow_select(
            num_clients_subset=num_clients_subset,
            candidate_set_size=int(selector_context["candidate_set_size"]),
            clients_probs=selector_context.get("clients_probs"),
            clients_losses=selector_context.get("clients_losses"),
            amount_of_clients=amount_of_clients,
            client_pool=client_pool,
            rng=rng,
            topk_tau=pow_topk_tau,
        )

    if selector_name == FEDCOR_SELECTOR:
        return fedcor_select(
            num_clients_subset=num_clients_subset,
            amount_of_clients=amount_of_clients,
            cur_round=int(selector_context["cur_round"]),
            warmup=int(selector_context["warmup"]),
            covariance=selector_context.get("covariance"),
            discount=selector_context.get("discount"),
            ts=selector_context.get("ts"),
            client_pool=client_pool,
            rng=rng,
            tau=fedcor_tau,
        )

    if selector_name == OORT_SELECTOR:
        return oort_select(
            num_clients_subset=num_clients_subset,
            training_cl_losses=selector_context["training_cl_losses"],
            last_participation_round=selector_context["last_participation_round"],
            round_times=selector_context["round_times"],
            cur_round=int(selector_context["cur_round"]),
            alpha_stale=float(selector_context["alpha_stale"]),
            loss_prior=float(selector_context["loss_prior"]),
            c=float(selector_context["c"]),
            preferred_time=float(selector_context["preferred_time"]),
            penalty_degree=float(selector_context["penalty_degree"]),
            amount_of_clients=amount_of_clients,
            client_pool=client_pool,
            rng=rng,
        )

    raise NotImplementedError(f"Unsupported selector_name={selector_name}")
