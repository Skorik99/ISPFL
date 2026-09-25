from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)

        self.load_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fpath = self.df.at[index, "fpath"]
        img = Image.open(fpath)
        img_tensor = self.load_transform(img)
        label = self.df.at[index, "target"]
        return index, ([img_tensor], label)
