import ast
import torch
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, df, cfg, mode="train"):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.pad_token_id = cfg.dataset.get("pad_token_id", 0)
        self.ignore_index = cfg.dataset.get("ignore_index", -100)
        self.examples = self._load_examples()

    def _load_examples(self):
        examples = []
        for _, row in self.df.iterrows():
            inputs = list(ast.literal_eval(row["x"]))
            targets = row["target"]

            # Ensure targets align with inputs length (for seq-to-seq next-token prediction)
            if len(targets) != len(inputs):
                # shift inputs by one and reuse last token to keep length
                targets = inputs[1:] + [inputs[-1]]
            examples.append((inputs, targets))
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        inputs, targets = self.examples[index]
        input_tensor = torch.tensor(inputs, dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        return index, ([input_tensor], target_tensor)
