import torch
from torch.utils.data import ConcatDataset, Subset, random_split

import os
import time
import random

import pickle

from torchvision.datasets import ImageFolder

from config_loader import *
from lidar_preprocessing import preprocess_image


class LidarDataset(ImageFolder):

    # Source: https://github.com/pytorch/vision/issues/2930
    # This creates a dataset with the same functionality as ImageFolder but with our custom targets.
    def __init__(self, name: str) -> None:

        input_folder = os.path.join(DATASET_FOLDER, name)

        with open(os.path.join(DATASET_FOLDER, name, "targets.pickle"), "rb") as file:
            labels = pickle.load(file)

        super().__init__(input_folder, preprocess_image)
        paths, _ = zip(*self.imgs)
        self.targets = [torch.Tensor(labels[path]) for path in paths]
        self.samples = self.imgs = list(zip(paths, self.targets))

    def get_idx_of_turn_samples(self):
        idx = []
        for i, x in enumerate(self.targets):
            if abs(x[0]) > 0.15:
                idx.append(i)
            elif abs(x[0]) > 0.05 and random.random() < 0.3:
                idx.append(i)
            elif random.random() < 0.05:
                idx.append(i)

        return idx


def display_dataset_example(ds: LidarDataset):
    """Displays a few example inputs from the given Lidar Dataset."""
    ds_item = ds.__getitem__(0)
    ds_item['input'].show()
    ds_item = ds.__getitem__(1000)
    ds_item['input'].show()
    ds_item = ds.__getitem__(2000)
    ds_item['input'].show()
    ds_item = ds.__getitem__(3000)
    ds_item['input'].show()
    ds_item = ds.__getitem__(4000)
    ds_item['input'].show()


print("Starting dataset loading...")
start_time = time.time()

datasets = []

for i, x in enumerate(DATASET_LIST):
    ds = LidarDataset(x)
    if DATASET_FILTER_ONLY_TURNS:
        ds = Subset(ds, ds.get_idx_of_turn_samples())
    datasets.append(ds)
    print("Dataset " + str(i) + "/" + str(len(DATASET_LIST)) + " loaded...")
    if DEBUG_GENERATE_ONLY_ONE_DATASET:
        print("DEBUG_REDUCED_DATASET enabled. Ignoring other datasets...")
        break

full_dataset = ConcatDataset(datasets)

print("Concatenating and generating splits...")

train_length = round(0.8*len(full_dataset))
[dataset_train, dataset_test] = random_split(full_dataset, [train_length, len(full_dataset) - train_length])

print("Dataset loading complete!")
print("Loading time: %s" % (time.time() - start_time))
