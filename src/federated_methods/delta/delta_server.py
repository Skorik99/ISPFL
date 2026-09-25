from ..fedavg.fedavg_server import FedAvgServer
import numpy as np
import torch


class DeltaServer(FedAvgServer):
    def __init__(self, cfg, alpha_1, alpha_2):
        super().__init__(cfg)

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        self.client_probs = np.array(
            [1.0 / self.amount_of_clients for _ in range(self.amount_of_clients)]
        )
        print(f"Initial clients probabilities:\n{self.client_probs}")

        self.client_sigmas = [None for _ in range(self.amount_of_clients)]

    def update_probs(self, participated_clients):
        self.no_buffer_state_dict = [
            name
            for name, param in self.global_model.named_parameters()
            if param.requires_grad
        ]

        N = len(participated_clients)
        mean_update = {
            param_name: torch.zeros_like(
                self.client_gradients[participated_clients[0]][param_name]
            )
            for param_name in self.no_buffer_state_dict
        }
        for rank in participated_clients:
            for param_name in self.no_buffer_state_dict:
                mean_update[param_name].add_(
                    self.client_gradients[rank][param_name]
                )
        for param_name in self.no_buffer_state_dict:
            mean_update[param_name].mul_(1.0 / N)

        client_scores = {}
        for rank in participated_clients:
            squared_distance = sum(
                (
                    self.client_gradients[rank][param_name]
                    - mean_update[param_name]
                )
                .square()
                .sum()
                for param_name in self.no_buffer_state_dict
            )
            client_scores[rank] = np.sqrt(
                self.alpha_1 * squared_distance.item()
                + self.alpha_2 * self.client_sigmas[rank] ** 2
            )

        new_probs = self.client_probs.copy()
        participated = set(participated_clients)
        unchanged_probability = sum(
            probability
            for rank, probability in enumerate(self.client_probs)
            if rank not in participated
        )
        score_sum = sum(client_scores.values())
        for rank in participated_clients:
            new_probs[rank] = (
                client_scores[rank]
                / score_sum
                * (1 - unchanged_probability)
            )

        return new_probs

    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        self.client_sigmas[client_result["rank"]] = client_result["sigma"]

    def select_clients_to_train(self, num_clients_subset, server_sampling=False):
        if num_clients_subset == self.amount_of_clients:
            return list(range(self.amount_of_clients))

        clients = list(range(self.amount_of_clients))
        selected_clients = np.random.choice(
            clients, size=num_clients_subset, replace=False, p=self.client_probs
        ).tolist()

        return selected_clients
