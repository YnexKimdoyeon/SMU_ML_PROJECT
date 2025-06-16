from torch.utils.data import Dataset
from PIL import Image
import os

class PillDataset(Dataset):
    def __init__(self, df, img_dir, label2id, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label2id = label2id
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = self.label2id[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label
