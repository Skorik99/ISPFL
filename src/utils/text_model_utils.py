import torch
import torch.nn as nn


class ShakespeareModel(nn.Module):
    """
    LSTM-based language model following the LEAF Shakespeare spec:
    2-layer LSTM with 256 hidden units per layer, 8-dim character embeddings.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=8,
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
        pad_token_id=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # Xavier init for embeddings and output for stable start
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)
        lstm_output = self.dropout(lstm_output)
        logits = self.output(lstm_output)
        return logits
