import math
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    fbeta_score,
)

from .cifar_utils import calculate_cifar_metrics


__all__ = [
    "stopping_criterion",
    "gpu_calculate_metrics",
]


def stopping_criterion(
    val_loss,
    metrics,
    best_metrics,
    epochs_no_improve,
):
    """
    Define stopping criterion for metrics from config['saving_metrics']
    best_metrics is updated only if every metric from best_metrics.keys() has improved

    :param val_loss: validation loss
    :param metrics: validation metrics
    :param best_metrics: the best metrics for the current epoch
    :param epochs_no_improve: number of epochs without best_metrics updating

    :return: epochs_no_improve, best_metrics
    """
    # get average metrics by class
    metrics = dict(metrics.mean(axis=1))
    # define condition best_metric >= metric for all except for loss
    metrics_mask = all(
        metrics[key] >= best_metrics[key] for key in best_metrics.keys() - {"loss"}
    )
    if not metrics_mask:
        epochs_no_improve += 1
        return epochs_no_improve, best_metrics
    if "loss" in list(best_metrics.keys()):
        if val_loss >= best_metrics["loss"]:
            epochs_no_improve += 1
            return epochs_no_improve, best_metrics
    # Updating best_metrics
    for key in list(best_metrics.keys()):
        if key == "loss":
            best_metrics[key] = val_loss
        else:
            best_metrics[key] = metrics[key]
    # Updating epochs_no_improve
    epochs_no_improve = 0
    return epochs_no_improve, best_metrics


def calculate_metrics(
    fin_targets,
    fin_outputs,
    verbose=False,
):
    # Get results
    softmax = torch.nn.Softmax(dim=1)
    results = softmax(torch.as_tensor(fin_outputs)).max(dim=1)[1]
    fin_targets = torch.as_tensor(fin_targets)
    # Calc metrics
    metrics = calculate_cifar_metrics(fin_targets, results, verbose)
    prediction_threshold = None
    return metrics, prediction_threshold


def gpu_calculate_metrics(fin_targets, fin_predictions, verbose=False):
    num_classes = int(
        torch.maximum(fin_targets.max(), fin_predictions.max()).item()
    ) + 1
    class_indices = fin_targets * num_classes + fin_predictions
    confusion = torch.bincount(
        class_indices, minlength=num_classes**2,
    ).reshape(num_classes, num_classes).to(torch.float32)

    true_positives = confusion.diagonal()
    predicted_counts = confusion.sum(dim=0)
    target_counts = confusion.sum(dim=1)
    active_classes = (predicted_counts + target_counts) > 0
    precision = torch.where(
        predicted_counts > 0,
        true_positives / predicted_counts,
        torch.zeros_like(true_positives),
    )
    recall = torch.where(
        target_counts > 0,
        true_positives / target_counts,
        torch.zeros_like(true_positives),
    )
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    metric_values = torch.stack(
        [
            true_positives.sum() / confusion.sum(),
            precision[active_classes].mean(),
            recall[active_classes].mean(),
            f1[active_classes].mean(),
        ]
    ).cpu().tolist()
    metrics = pd.DataFrame(
        {"cifar": metric_values},
        index=["Accuracy", "Precision", "Recall", "f1-score"],
    )
    if verbose:
        print(metrics)
    return metrics, None


def check_metrics_names(metrics):
    allowed_metrics = [
        "loss",
        "Specificity",
        "Sensitivity",
        "G-mean",
        "f1-score",
        "fbeta2-score",
        "ROC-AUC",
        "AP",
        "Precision (PPV)",
        "NPV",
    ]

    assert all(
        [k in allowed_metrics for k in metrics.keys()]
    ), f"federated_params.server_saving_metrics can be only {allowed_metrics}, but get {metrics.keys()}"
