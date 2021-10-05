import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from random import shuffle
import pandas as pd


class PixWiseDataset():
    def __init__(self, csvfile, map_size=14,
                 smoothing=True, transform=None):
        self.data = pd.read_csv(csvfile)
        self.transform = transform
        self.map_size = map_size
        self.label_weight = 0.99 if smoothing else 1.0

    def dataset(self):
        images = []
        labels = []
        masks = []

        for ind in self.data.index:
            img_name = self.data.iloc[ind]['name']
            img = Image.open(img_name)
            # img = cv.resize(img, (224, 224))
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # img = np.moveaxis(img, 2, 0)
            # img = np.asarray(img)

            label = self.data.iloc[ind]['label']
            if label == 0:
                mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1 - self.label_weight)
            else:
                mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (self.label_weight)

            if self.transform:
                img = self.transform(img)

            images.append(img)
            labels.append(label)
            masks.append(mask)

        labels = np.array(labels, dtype=np.float32)

        dataset = [[images[i], masks[i], labels[i]] for i in range(len(images))]
        return dataset
