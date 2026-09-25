from pandas import cut
from ..fedavg.fedavg_server import FedAvgServer
import numpy as np


class OortServer(FedAvgServer):
    def __init__(self, cfg, slow_clients_count):
        super().__init__(cfg)
        self.training_cl_losses = [None] * self.amount_of_clients

        # last participation tracker
        self.last_participation_round = [0] * self.amount_of_clients

        self.round_times = [None] * self.amount_of_clients

        # EMA smoothing
        self.alpha_smooth = 0.5
        # exploitation cutoff epsilon (1-eps)*K
        self.eps = 0.05
        # staleness term scale (UCB-like)
        self.alpha_stale = 10.0
        # small prior for clients without loss
        self.loss_prior = 1e-6
        # C %
        self.c = 0.6
        # params for system heterogenity
        self.preffered_time = 14
        self.penalty_degree = 1

        # synthetic slow client computational time params
        self.amount_clients_to_slow = slow_clients_count
        self.p_additional_time = 0.1
        self.slowed_clients = [None] * self.amount_of_clients

    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        rank = client_result["rank"]
        if rank in self.list_clients:
            loss = client_result["training_loss"]

            cur_loss = self.training_cl_losses[rank]

            # EMA smoothing
            if cur_loss is None:
                new_loss = loss
            else:
                new_loss = self.alpha_smooth * cur_loss + (1 - self.alpha_smooth) * loss

            self.training_cl_losses[rank] = new_loss

            self.last_participation_round[rank] = self.cur_round
            
            client_time = client_result["time"]
            if rank in self.slowed_clients:
                client_time += np.random.geometric(self.p_additional_time)
            
            self.round_times[rank] = client_time

    def select_clients_to_train(self, num_clients_subset, server_sampling=False):
        T = self.cur_round + 1  # avoid ln(0)

        # === Step 1: base utility U_i (loss) ===
        U = np.array(
            [
                loss if loss is not None else self.loss_prior
                for loss in self.training_cl_losses
            ]
        )

        sys_coef = [
            (
                (self.preffered_time / time) ** self.penalty_degree
                if ((time is not None) and (time < self.preffered_time))
                else 1
            )
            for time in self.round_times
        ]
        U = U * np.array(sys_coef)

        # === Step 2: staleness incentive term ===
        # I_i = alpha * sqrt( ln(T) / (last_round[i] + 1) )
        stale = np.array(
            [self.last_participation_round[i] for i in range(self.amount_of_clients)]
        )
        stale = np.maximum(stale, 1)  # avoid division by zero
        incentive = self.alpha_stale * np.sqrt(np.log(T) / stale)

        U_tilde = U + incentive

        # === Step 3: clipping at 95-percentile ===
        U95 = np.percentile(U_tilde, 95)
        U_tilde = np.minimum(U_tilde, U95)

        # === Step 4: sort utilities ===
        sort_idx = np.argsort(-U_tilde)  # descending

        # position q = (1 - eps) * K
        q = num_clients_subset - 1  # int(max(1, (1 - self.eps) * num_clients_subset))
        # if q > len(sort_idx) - 1:
        #     q = len(sort_idx) - 1
        Uq = U_tilde[sort_idx[q]]

        # cutoff with c%
        cutoff = self.c * Uq

        # === Step 5: High Utility Pool ===
        pool_mask = U_tilde >= cutoff
        pool_indices = np.where(pool_mask)[0]

        # === Step 6: probabilistic sampling among pool ===
        pool_utils = U_tilde[pool_indices]
        probs = pool_utils / pool_utils.sum()

        # sample K clients without replacement
        chosen = np.random.choice(
            pool_indices, size=num_clients_subset, replace=False, p=probs
        )

        # Select random client to slow
        self.slowed_clients = np.random.choice(
            chosen.tolist(),
            size=min(self.amount_clients_to_slow, len(chosen)),
            replace=False,
        )
        return chosen.tolist()
