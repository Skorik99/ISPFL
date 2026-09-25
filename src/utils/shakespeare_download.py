import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from datasets import load_dataset

RANDOM_STATE = 42
TEST_PROP = 0.1
TRUST_PROP = 0.05

CLIENT_COLUMN = "character_id"
INPUT_COLUMN = "x"
TARGET_COLUMN = "y"


def _to_int_tokens(seq):
    return [ord(ch) for ch in seq]


def _normalize_clients(df, client_col):
    unique_clients = sorted(df[client_col].unique())
    client_map = {client: idx + 1 for idx, client in enumerate(unique_clients)}
    df["client"] = df[client_col].map(client_map)
    return df, client_map


def _split_client_df(df: pd.DataFrame, test_prop: float, trust_prop: float):
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    total = len(df)
    if total < 3:
        return df, pd.DataFrame(), pd.DataFrame()

    test_n = int(total * test_prop)
    trust_n = int(total * trust_prop)
    if test_n + trust_n >= total:
        test_n = max(1, min(total - 1, test_n))
        trust_n = max(0, total - test_n - 1)

    test_df = df.iloc[:test_n]
    trust_df = df.iloc[test_n : test_n + trust_n]
    train_df = df.iloc[test_n + trust_n :]
    return train_df, test_df, trust_df


def _split_by_client(df: pd.DataFrame, test_prop: float, trust_prop: float):
    train_parts = []
    test_parts = []
    trust_parts = []

    for client_id, client_df in df.groupby("client"):
        client_train, client_test, client_trust = _split_client_df(
            client_df, test_prop, trust_prop
        )
        train_parts.append(client_train)
        test_parts.append(client_test)
        trust_parts.append(client_trust)

    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
        pd.concat(trust_parts).reset_index(drop=True),
    )


def _save_split(df, split_name, save_path, input_col, target_col):
    os.makedirs(save_path, exist_ok=True)

    data = []
    max_token = 0
    for _, row in df.iterrows():
        inputs = _to_int_tokens(row[input_col])
        targets = _to_int_tokens(row[target_col])
        if len(inputs) == 0 or len(targets) == 0:
            continue

        max_token = max(
            max_token,
            max(inputs),
            max(targets),
        )

        data.append(
            {
                "x": inputs,
                "target": targets,
                "client": row["client"],
                "sequence_length": len(inputs),
            }
        )

    df = pd.DataFrame(data)
    csv_path = os.path.join(save_path, f"{split_name}_df.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, max_token, df["sequence_length"].max() if len(df) > 0 else 0


def _update_config(
    config_dir,
    train_path: Dict[str, str],
    test_path: str,
    vocab_size,
    pad_token_id,
    max_sequence_length,
):
    config_names = ["shakespeare.yaml", "shakespeare_trust.yaml"]
    for filename in config_names:
        filepath = os.path.join(config_dir, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        data_sources = data.get("data_sources", {})
        if "test_directories" in data_sources and test_path is not None:
            data_sources["test_directories"] = [test_path]
        if "train_directories" in data_sources:
            if filename == "shakespeare_trust.yaml":
                data_sources["train_directories"] = [train_path["trust"]]
            else:
                data_sources["train_directories"] = [train_path["train"]]

        data["vocab_size"] = vocab_size
        data["pad_token_id"] = pad_token_id
        data["max_sequence_length"] = max_sequence_length
        data["data_sources"] = data_sources

        with open(filepath, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)


def process_shakespeare(target_dir="shakespeare_data"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "..", "configs", "observed_data_params")
    save_path = os.path.abspath(os.path.join(target_dir, "shakespeare_data"))

    print("Downloading Shakespeare...", flush=True)
    dataset = load_dataset("flwrlabs/shakespeare")

    if "train" not in dataset:
        raise ValueError("Shakespeare dataset must provide a 'train' split")

    df = dataset["train"].to_pandas()

    client_col = CLIENT_COLUMN
    input_col = INPUT_COLUMN
    target_col = TARGET_COLUMN

    df, client_map = _normalize_clients(df, client_col)
    print(f"Found {len(client_map)} clients")

    train_df, test_df, trust_df = _split_by_client(
        df, test_prop=TEST_PROP, trust_prop=TRUST_PROP
    )

    print("Converting Shakespeare splits to map files (x/target per-row)...")
    train_path, max_train_token, max_train_seq = _save_split(
        train_df, "train", save_path, input_col, target_col
    )
    trust_path, max_trust_token, max_trust_seq = _save_split(
        trust_df, "trust", save_path, input_col, target_col
    )
    test_path, max_test_token, max_test_seq = _save_split(
        test_df, "test", save_path, input_col, target_col
    )

    if os.path.getsize(train_path) == 0 or os.path.getsize(test_path) == 0:
        raise ValueError("Processed dataset splits are empty. Please check input data.")

    max_token = max(max_train_token, max_trust_token, max_test_token)
    max_sequence_length = max(1, int(max(max_train_seq, max_trust_seq, max_test_seq)))
    vocab_size = max_token + 2  # Add padding token
    pad_token_id = vocab_size - 1

    print("Updating observed_data_params configs...")
    _update_config(
        config_dir,
        {"train": train_path, "trust": trust_path},
        test_path,
        vocab_size,
        pad_token_id,
        max_sequence_length,
    )

    print("All steps completed successfully!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Shakespeare dataset for federated training."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=".",
        help="Directory to save processed files (default: current directory)",
    )
    args = parser.parse_args()

    process_shakespeare(args.target_dir)
