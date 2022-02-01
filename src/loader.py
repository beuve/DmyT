from pathlib import Path
import numpy as np
import torch
import os
from PIL import Image
import torchvision.transforms as transforms


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, directory, transforms=None, shuffle=True):
        self.labels = os.listdir(directory)
        self.transforms = transforms
        self.paths = []
        for root, _, files in os.walk(directory, topdown=False):
            l = DataLoader.find_label(root, self.labels, directory)
            for name in files:
                file = os.path.join(root, name)
                if file.endswith(".jpg") or file.endswith(
                        ".png") or file.endswith(".jpeg"):
                    self.paths.append((file, l))
        if shuffle:
            np.random.shuffle(self.paths)
        print(f"found {len(self.paths)} file from {len(self.labels)}")

    def find_label(path, labels, root):
        for i, l in enumerate(labels):
            p = Path(path)
            _root = Path(os.path.join(root, l))
            if _root in (p, *p.parents):
                return i

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path, label = self.paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        l = [0] * len(self.labels)
        l[label] = 1
        if (len(l) == 2):
            l = [1 - label for label in l]
        l = torch.tensor(l)
        assert l.sum() == 1
        return img, l


def loader(path, img_size, batch_size=1, split='test', shuffle=True):

    transforms_list = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = DataLoader(
        path + "/" + split,
        transforms=transforms_list,
        shuffle=shuffle,
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=2,
    )
    return loader