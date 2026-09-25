import torch
import numpy as np


def get_loss(loss_cfg, df=None, device=None, init_pos_weight=None):
    loss_name = loss_cfg.loss_name
    loss = loss_cfg.config
    if loss_name == "ce":
        if init_pos_weight:
            pos_weight = calculate_class_weights_multi_class(df)
            pos_weight = torch.tensor(pos_weight).to(device)
        else:
            pos_weight = None
        return torch.nn.CrossEntropyLoss(
            weight=pos_weight,
            ignore_index=loss.ignore_index,
            reduction=loss.reduction,
            label_smoothing=loss.label_smoothing,
        )
    else:
        raise ValueError("Unknown type of loss function")


def calculate_class_weights_multi_class(df):
    df_copy = df.copy()
    target_array = np.array(df_copy["target"].tolist())

    class_weights = {}
    ordered_weights = []

    unique_classes = np.unique(target_array)
    total_count = len(target_array)

    for cls in unique_classes:
        class_count = np.sum(target_array == cls, axis=0)
        class_weight = float(total_count / (len(unique_classes) * class_count))
        class_weights[cls] = class_weight

    # Ensure the weights are added to the list in order of ascending class index
    for cls in sorted(unique_classes):
        ordered_weights.append(class_weights[cls])

    return ordered_weights
