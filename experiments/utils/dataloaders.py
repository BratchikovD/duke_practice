from torchvision import datasets
import torch
import numpy as np


class TripletImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(TripletImageFolder, self).__init__(root, transform=transform)

    def __getitem__(self, index):
        path, label = self.imgs[index]
        anchor_image = self.loader(path)

        # Получаем индексы положительных и отрицательных примеров
        pos_indices = np.where(np.array(self.targets) == label)[0]
        neg_indices = np.where(np.array(self.targets) != label)[0]

        pos_index = np.random.choice(pos_indices)
        neg_index = np.random.choice(neg_indices)

        pos_path, _ = self.imgs[pos_index]
        neg_path, _ = self.imgs[neg_index]

        pos_image = self.loader(pos_path)
        neg_image = self.loader(neg_path)

        if self.transform is not None:
            anchor_image = self.transform(anchor_image)
            pos_image = self.transform(pos_image)
            neg_image = self.transform(neg_image)

        return anchor_image, pos_image, neg_image, label

    def __len__(self):
        return len(self.imgs)
