import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from omegaconf import open_dict

from .image_net_utils import ImageNetDataset
from .cifar_utils import ImageDataset, get_image_dataset_params
from .shakespeare_utils import ShakespeareDataset


def prepare_df_for_federated_training(
    cfg: dict,
    directories_key: str,
):
    df = read_dataframe_from_cfg(cfg, directories_key)
    uses_dirichlet_partition = "dirichlet" in cfg.dataset.data_name

    # Shakespeare version to select subset of clients
    amount_clients_subset = getattr(cfg.dataset, "amount_of_clients", None)
    if amount_clients_subset is not None:
        if "client" not in df.columns:
            raise ValueError(
                "Client subsampling requires an input CSV with a 'client' column."
            )
        unique_clients = df["client"].unique()
        rng = np.random.default_rng(cfg.random_state)
        chosen = rng.choice(unique_clients, size=amount_clients_subset, replace=False)
        allowed_clients = sorted(set(chosen))
        df = df[df["client"].isin(allowed_clients)]
        df["client"] = df["client"].apply(lambda x: allowed_clients.index(x) + 1)

    if uses_dirichlet_partition:
        # Dirichlet datasets assign clients from labels, so source maps do not
        # need to carry a pre-existing client assignment.
        df = create_dirichlet_df(df, cfg)
    else:
        if "client" not in df.columns:
            raise ValueError(
                "Federated data must provide a 'client' column unless the "
                "dataset uses a Dirichlet partition."
            )
        df["client"] = df["client"].apply(lambda x: x - 1)

    df.reset_index(drop=True, inplace=True)

    # Text datasets can provide vocab size explicitly to avoid parsing sequence columns
    if hasattr(cfg.dataset, "vocab_size") and cfg.dataset.vocab_size is not None:
        num_classes = cfg.dataset.vocab_size
    else:
        num_classes = df['target'].nunique()
    
    print(f"Num classes = {num_classes}")

    with open_dict(cfg):
        cfg.training_params.num_classes = num_classes

    print("Preprocess successfull\n")
    return df, cfg


def read_dataframe_from_cfg(
    cfg,
    directories_key="train_directories",
    mode="dataset",
):
    df = pd.DataFrame()
    for directories in cfg[mode]["data_sources"][directories_key]:
        df = pd.concat([df, pd.read_csv(directories, low_memory=False)])
    return df


def get_dataset_loader(
    df: pd.DataFrame,
    cfg,
    drop_last=True,
    mode="train",
    unsupervised=False,
    transforms=None,
):
    if "cifar" in cfg["dataset"]["data_name"]:
        # CIFAR case
        image_size, mean, std = get_image_dataset_params(cfg, df)
        dataset = ImageDataset(df, mode, image_size, mean, std)
    elif "shakespeare" in cfg["dataset"]["data_name"]:
        dataset = ShakespeareDataset(df, cfg, mode)
    else:
        # ImageNet case
        dataset = ImageNetDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.training_params.num_workers,
        drop_last=drop_last,
    )
    assert (
        len(loader) > 0
    ), f"len(dataloader) is 0, either lower the batch size, or put drop_last=False"
    return loader


def get_stratified_subsample(df, num_samples, random_state):
    """Create a subDataFrame with `num_samples` and stratified label distribution

    Args:
        df (pd.DataFrame): origin DataFrame
        num_samples (_type_): number of samples in subDataFrame

    return: sub_df (pd.DataFrame): sub DataFrame
    """
    sub_df = pd.DataFrame()
    for target in list(df.target.value_counts().keys()):
        tmp = df[df["target"] == target]
        weight = len(tmp) / len(df)
        amount = int(weight * num_samples)
        sub_df = pd.concat(
            [
                sub_df,
                tmp.sample(
                    n=amount,
                    random_state=random_state,
                ),
            ]
        )
    # Remove all rows from df, that are now in sub_df
    df = df[~df["fpath"].isin(list(sub_df["fpath"]))]
    return df, sub_df


def dirichlet_distrubution(
    total_data_points, num_classes, num_clients, alpha, verbose, seed=42
):
    np.random.seed(seed)
    dirichet = np.random.dirichlet(alpha * np.ones(num_clients), num_classes)
    data_distr = (dirichet * total_data_points / num_classes).astype(int)
    data_distr = data_distr.transpose()

    total_assigned = data_distr.sum()
    remaining_data_points = total_data_points - total_assigned
    max_per_class = total_data_points // num_classes

    class_counts = {i: data_distr[:, i].sum() for i in range(num_classes)}

    # Distribute remaining data (because we use .astype(int))
    if remaining_data_points > 0:
        for i in range(remaining_data_points):
            for class_idx in range(num_classes):
                if class_counts[class_idx] < max_per_class:
                    client_idx = np.argmin(data_distr.sum(axis=1))
                    data_distr[client_idx, class_idx] += 1
                    class_counts[class_idx] += 1
                    break

    if verbose:
        print("Total usage data:", data_distr.sum())
        for i, x in enumerate(data_distr):
            x_str = " ".join(f"{num:>4}" for num in x)
            print(f"Client {i:>2} | {x_str} | {sum(x):>5}")

    return data_distr


def create_dirichlet_df(df, cfg):
    n_classes = df["target"].nunique()
    data_distr = dirichlet_distrubution(
        len(df),
        n_classes,
        cfg.federated_params.amount_of_clients,
        cfg.dataset.alpha,
        cfg.dataset.verbose,
        cfg.random_state,
    )

    # Drop 'client' column
    df["client"] = -1

    client_target_count = {
        i: {j: count for j, count in enumerate(row)} for i, row in enumerate(data_distr)
    }

    # Fill 'client' column
    for index, row in df.iterrows():
        target = row["target"]
        for client, counts in client_target_count.items():
            if counts[target] > 0:
                df.at[index, "client"] = client
                client_target_count[client][target] -= 1
                break

    # Check the results
    result = df.groupby(["client", "target"]).size().unstack(fill_value=0)
    for client, row in result.iterrows():
        if client != -1:
            expected = data_distr[client]
            actual = row.tolist()
            assert np.all(
                expected == actual
            ), f"Mismatch for client {client}: Expected {expected}, Actual {actual}"

    print("\nChecking: All clients have the correct distribution of targets.\n")

    return df


def print_df_distribution(df, num_classes, num_clients, pathology_names=None):
    is_multilabel = isinstance(df["target"].iloc[0], list)
    if is_multilabel:
        df["class_target"] = df["target"].apply(
            lambda x: [i for i, val in enumerate(x) if val == 1] or [-1]
        )

    print(f"Total usage data: {len(df[df['client'] != -1])}")

    client_groups = df.groupby("client")
    valid_clients = set(df["client"].unique())
    total_distribution = [0] * num_classes
    if is_multilabel:
        total_distribution = [0] * (num_classes + 1)

    for cl in range(num_clients):
        if cl not in valid_clients:
            print(f"Client {cl:>2} | No data")
            continue

        client_data = client_groups.get_group(cl)
        if is_multilabel:
            distr = (
                client_data["class_target"]
                .explode()
                .value_counts()
                .reindex(range(-1, num_classes), fill_value=0)
                .tolist()
            )
        else:
            distr = (
                client_data["target"]
                .value_counts()
                .reindex(range(num_classes), fill_value=0)
                .tolist()
            )
        x_str = " ".join(f"{num:>4}" for num in distr)

        total_distribution = [
            total + distr_val for total, distr_val in zip(total_distribution, distr)
        ]

        if cl == 0 and is_multilabel:
            pathology_str = " " * 11 + "  ".join(["Other"] + pathology_names)
            print(pathology_str)
        print(f"Client {cl:>2} | {x_str} | {len(client_data):>5}")

    total_x_str = " ".join(f"{num:>4}" for num in total_distribution)
    print(f"Total distribution | {total_x_str}")
