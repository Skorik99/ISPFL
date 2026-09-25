import cv2
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def load_image(image_path: str, windowing: bool = False):  # move somewhere else
    """
    Load an image from a file using OpenCV or DICOM format.

    Parameters
    ----------
    image_path:
        The path to the image file.
    windowing:
        The flag to apply windowing to DICOM images.

    Returns
    -------
    numpy.ndarray:
        The loaded image as a NumPy array.
        If the image is in DICOM format, it is normalized with values in the range [0, 1]
        and converted to RGB and normalized.
        If the image is in a common image format (e.g., JPEG, PNG), it is loaded and
        color channels are rearranged to RGB.

    Note
    ----
    This function supports loading both standard image formats and DICOM medical
    images. For DICOM images, it assumes that the pixel data is in Hounsfield units
    and normalizes it to the [0, 1] range.
    """

    if image_path.endswith(".dcm") or image_path.endswith(".dicom"):
        raise NotImplementedError("DICOM images are not supported yet.")
    else:
        image = cv2.imread(image_path).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image /= 255.0
    return image


def get_image_dataset_params(cfg, df):
    if "cifar" in cfg.dataset.data_name:
        image_size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif "stl10" in cfg.dataset.data_name:
        image_size = 96
        mean = (0.3881, 0.4287, 0.4413)
        std = (0.2237, 0.226, 0.2305)
    else:
        warnings.warn(
            "Only Cifar and STL-10 are supported, you should add a mean, standard deviation \
            and image size into utils.cifar_utils.get_image_dataset_params for your dataset"
        )
        mean = np.zeros(3)
        std = np.zeros(3)
        for _, row in df.iterrows():
            image = load_image(row["fpath"])
            image = image.T.reshape(image.shape[0], -1)
            mean += np.mean(image, axis=1)
            std += np.std(image, axis=1)
        mean = mean[::-1] / len(df)
        std = std[::-1] / len(df)
        image_size = load_image(row["fpath"]).shape[0]
    return image_size, mean, std


class ImageDataset(Dataset):
    def __init__(self, df, transform_mode, image_size, mean, std):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transform = self.set_up_transform(transform_mode)
        self.mode = transform_mode

    def __len__(self):
        return len(self.df)

    def set_up_transform(self, mode):
        if mode == "train":
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
        elif mode == "trust_corrupted":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomGrayscale(p=0.8),
                    transforms.ColorJitter(
                        brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.3,
                    ),
                    transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                        p=0.3,
                    ),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
        return transform

    def __getitem__(self, index):
        image = Image.open(self.df["fpath"][index])
        image = self.transform(image)
        label = self.df["target"][index]
        return index, ([image], label)


def calculate_cifar_metrics(fin_targets, results, verbose=False):
    df = pd.DataFrame(
        columns=["cifar"],
        index=[
            "Accuracy",
            "Precision",
            "Recall",
            "f1-score",
        ],
    )
    df.loc["Accuracy", "cifar"] = accuracy_score(fin_targets, results)
    df.loc["Precision", "cifar"] = precision_score(
        fin_targets, results, average="macro", zero_division=0
    )
    df.loc["Recall", "cifar"] = recall_score(
        fin_targets, results, average="macro", zero_division=0
    )
    df.loc["f1-score", "cifar"] = f1_score(
        fin_targets, results, average="macro", zero_division=0
    )
    if verbose:
        print(df)
    return df
