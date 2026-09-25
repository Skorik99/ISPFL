import os
import shutil
import tarfile
import requests
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import argparse


# Set global seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def download_cifar10(target_dir="cifar10"):
    os.makedirs(target_dir, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(target_dir, "cifar-10-python.tar.gz")

    if not os.path.exists(tar_path):
        print("Downloading CIFAR-10...")
        response = requests.get(url, stream=True)
        with open(tar_path, "wb") as f:
            f.write(response.content)

    # Extract files
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    os.remove(tar_path)


def process_cifar10(base_dir="cifar10"):
    img_dir = os.path.join(base_dir, "images", "data")
    os.makedirs(img_dir, exist_ok=True)

    if not os.path.isabs(img_dir):
        curent_run_path = os.getcwd()
        img_dir = os.path.join(curent_run_path, img_dir)

    print("Converting CIFAR-10...")
    for split in ["train", "test"]:
        files = (
            ["data_batch_%d" % i for i in range(1, 6)]
            if split == "train"
            else ["test_batch"]
        )

        for file in files:
            with open(os.path.join(base_dir, "cifar-10-batches-py", file), "rb") as f:
                data = pickle.load(f, encoding="bytes")

            images = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            filenames = data[b"filenames"]

            for img, orig_filename in zip(images, filenames):
                path = os.path.join(img_dir, orig_filename.decode("utf-8"))
                Image.fromarray(img).save(path)

    shutil.rmtree(os.path.join(base_dir, "cifar-10-batches-py"))


def update_fpath(target_path):
    print("Setting paths to .csv files...")
    map_files_dir = os.path.join(os.getcwd(), "src/configs/observed_data_params/")

    image_path = os.path.join(target_path, "images", "data")

    if not os.path.isabs(image_path):
        curent_run_path = os.getcwd()
        image_path = os.path.join(curent_run_path, image_path)

    map_file_names = [
        "test_map_file.csv",
        "train_map_file.csv",
        "trust_map_file.csv",
    ]

    for map_file in map_file_names:
        map_file_path = os.path.join(map_files_dir, map_file)

        df = pd.read_csv(map_file_path, index_col=False)
        df["fpath"] = df["fpath"].apply(
            lambda path: os.path.join(image_path, os.path.basename(path))
        )

        df.to_csv(map_file_path, index=False)


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

    # # 1. Download dataset
    download_cifar10(args.target_dir)

    # 2. Convert to images, create a map-file
    process_cifar10(args.target_dir)

    # 3. Update fpath in .csv files
    update_fpath(args.target_dir)
