import os
import yaml
import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def save_df(df, mode, save_path):
    data = []
    for idx, row in df.iterrows():
        image = row["image"]
        target = row["target"]

        transformed_img = image.convert("RGB")

        filename = f"{mode}_{idx}.png"
        file_path = os.path.join(save_path, filename)
        transformed_img.save(file_path)

        data.append({"fpath": file_path, "target": target, "client": -1})

    df = pd.DataFrame(data)
    return df


def get_dfs():
    print("Downloading TinyImageNet...")
    train_dataset = load_dataset("Maysee/tiny-imagenet", split="train")
    test_dataset = load_dataset("Maysee/tiny-imagenet", split="valid")

    print("Converting TinyImageNet...")
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)

    train_df, trust_df = train_test_split(
        train_df, test_size=0.04, stratify=train_df["label"]
    )

    train_df = train_df.rename(columns={"label": "target"})
    trust_df = trust_df.rename(columns={"label": "target"})
    test_df = test_df.rename(columns={"label": "target"})

    return train_df, trust_df, test_df


def image_net_processing(target_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "..", "configs", "observed_data_params")
    save_path = os.path.join(target_dir, "image_net_data")

    os.makedirs(save_path, exist_ok=True)

    train_df, trust_df, test_df = get_dfs()

    print("Saving TinyImageNet...")

    train_df = save_df(train_df, "train", save_path)
    trust_df = save_df(trust_df, "trust", save_path)
    test_df = save_df(test_df, "test", save_path)

    train_df.to_csv(os.path.join(save_path, "train_df.csv"))
    trust_df.to_csv(os.path.join(save_path, "trust_df.csv"))
    test_df.to_csv(os.path.join(save_path, "test_df.csv"))

    print("Setting paths to .yaml files...")

    config_names = [
        "image_net_trust.yaml",
        "image_net.yaml",
    ]

    for filename in os.listdir(config_dir):
        if filename not in config_names:
            continue

        filepath = os.path.join(config_dir, filename)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        data_sources = data.get("data_sources", {})

        if "test_directories" in data_sources:
            test_map_path = [os.path.join(save_path, "test_df.csv")]
            data_sources["test_directories"] = test_map_path

        if "train_directories" in data_sources:
            if filename == "image_net_trust.yaml":
                train_map_name = "trust_df.csv"
            elif filename == "image_net.yaml":
                train_map_name = "train_df.csv"
            else:
                train_map_name = None

            if train_map_name is not None:
                train_map_path = [os.path.join(save_path, train_map_name)]
                data_sources["train_directories"] = train_map_path

        data["data_sources"] = data_sources

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CIFAR-10 dataset and create splits."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=".",
        help="Directory to save processed files (default: current directory)",
    )
    args = parser.parse_args()

    image_net_processing(args.target_dir)
    print("All steps completed successfully!!!")
