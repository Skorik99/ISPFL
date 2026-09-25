from ..fedavg.fedavg_server import FedAvgServer
import numpy as np


class PowServer(FedAvgServer):
    def __init__(self, cfg, candidate_set_size, df):
        super().__init__(cfg)
        
        self.candidate_set_size = candidate_set_size
        self.df = df
        
        self.clients_probs = self.set_probs_to_choise_client()
        
        self.clients_losses = [np.inf for _ in range(self.amount_of_clients)]

        
    def select_clients_to_train(self, num_clients_subset, server_sampling=False):
        if num_clients_subset == self.amount_of_clients:
            return list(range(self.cfg.federated_params.amount_of_clients))
         
        # Randomly select a subset of clients from the total pool
        # based on their dataset size
        # (a.k.a. A in paper, |A| = d)
        candidate_clients_list = np.random.choice(
            range(self.amount_of_clients),
            size=self.candidate_set_size,
            replace=False,
            p=self.clients_probs,
        )
        candidate_clients_list = candidate_clients_list.tolist()
        
        if not server_sampling:
            print(f"Current clients losses: {self.clients_losses}", flush=True)
            print(f"Current candidate clients: {candidate_clients_list}", flush=True)

        # Sort the selected candidates by their loss values
        candidate_clients_list.sort(
            key=lambda client_rank: self.clients_losses[client_rank], reverse=True
        )

        if not server_sampling:
            print(f"Sorted candidate clients: {sorted(candidate_clients_list)}", flush=True)
            ranked_losses = [
                (idx, client, self.clients_losses[client])
                for idx, client in enumerate(candidate_clients_list)
            ]
            print(f"Ranked candidate losses: {ranked_losses}", flush=True)


        # Select the top `amount_of_clients` clients from the sorted list
        selected_clients = candidate_clients_list[: num_clients_subset]
        
        if not server_sampling:
            print(f"Selected clients: {selected_clients}", flush=True)

        return selected_clients

    def set_probs_to_choise_client(self):
        clients_df_len = [
            len(self.df[self.df["client"] == i]) for i in range(self.amount_of_clients)
        ]
        total_data_count = sum(clients_df_len)
        clients_probs = [
            client_len / total_data_count for client_len in clients_df_len
        ]
        print(f"Probabilities of client selection: {clients_probs}")

        return clients_probs

