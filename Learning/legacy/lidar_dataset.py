import torch
from torch.utils.data import Dataset, ConcatDataset, random_split

import os
import time

import pickle, json
from math import degrees

# Load config
cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "config.json"))
CONFIG = json.load(open(cfg_path, "r", encoding="utf8"))
DATASET_FOLDER = CONFIG['DATASET_FOLDER']
DEBUG_GENERATE_ONLY_ONE_DATASET = CONFIG['DEBUG_GENERATE_ONLY_ONE_DATASET']


class LidarDataset(Dataset):

    def __init__(self, name: str) -> None:

        self.history_length = 10

        path = os.path.join(DATASET_FOLDER, name + ".pickle")

        with open(path, "rb") as file:
            self.dataset = pickle.load(file)

        prev_velocity = 0

        for i, entry in enumerate(self.dataset):
            self.dataset[i], prev_velocity = self.preprocessor(entry, prev_velocity)


    def _restrict_lidar_fov(self, data: list) -> list:
        """IRL Lidar has a 270° FOV. To train the network with data of the same size, 
        restrict the FOV of the lidar data from the simulation. """

        n = len(data)
        angle_increment = 360 / n
        i = round(45 / angle_increment)
        j = n-i-1
        return data[i:j]

    def _classify(self, steering_angle, velocity):
        # Possible actions
        # 0 - Forward
        # 1 - Backward
        # 2 - Stop
        # 3 - Right
        # 4 - Left
        # 5 - Slightly right
        # 6 - Slightly left
        # 7 - Slowdown

        if abs(velocity) < 0.3:
            # Stop
            return 2
        if degrees(abs(steering_angle)) < 3:
            # Forward
            return 0
        if steering_angle > 0:
            if degrees(abs(steering_angle)) < 10:
                # Slightly left
                return 6
            else:
                # Left
                return 4
        else:
            if degrees(abs(steering_angle)) < 10:
                # Slightly right
                return 5
            else:
                # Right
                return 3
            
    def _output_to_tensor(self, output):
        tensor = torch.zeros(8)
        tensor[output] = 1
        return tensor

    def preprocessor(self, entry, prev_velocity):
        input = self._restrict_lidar_fov(entry[0])
        output = self._classify(entry[1][0], entry[1][1])
        return ([torch.Tensor(input), prev_velocity, self._output_to_tensor(output)], entry[1][1])

    def addHistory(self, index):
        lidar = []
        velocities = []
        lidar_length = self.dataset[index][0].shape[0]
        for i in range(index-self.history_length+1, index+1):
            lidar.append(self.dataset[i][0].reshape((lidar_length, 1)))
            velocities.append(self.dataset[i][1])

        lidar_tensor = torch.cat(lidar, dim=1)
        velocities_tensor = torch.Tensor(velocities)
        return [lidar_tensor, velocities_tensor, self.dataset[index][2]]

    def __getitem__(self, idx):
        return self.addHistory(idx+self.history_length-1)

    def __len__(self):
        return len(self.dataset)-self.history_length+1


def display_dataset_example(ds: LidarDataset):
    """Displays a few example inputs from the given Lidar Dataset."""
    ds_item = ds.__getitem__(0)
    print(ds_item)


print("Starting dataset loading...")
start_time = time.time()

berlin_dataset = LidarDataset("dataset_map_berlin")
print("Dataset 1/4 loaded...")

if not DEBUG_GENERATE_ONLY_ONE_DATASET:
    columbia_dataset = LidarDataset("dataset_map_columbia")
    print("Dataset 2/4 loaded...")

    levine_dataset = LidarDataset("dataset_map_levine_blocked")
    print("Dataset 3/4 loaded...")

    mtl_dataset = LidarDataset("dataset_map_mtl")
    print("Dataset 4/4 loaded...")

    full_dataset = ConcatDataset(
        [berlin_dataset, columbia_dataset, levine_dataset, mtl_dataset])
else:
    print("DEBUG_REDUCED_DATASET enabled. Ignoring other datasets...")
    full_dataset = ConcatDataset([berlin_dataset])

print("Concatenating and generating splits...")

train_length = round(0.8*len(full_dataset))
[dataset_train, dataset_test] = random_split(
    full_dataset, [train_length, len(full_dataset) - train_length])

print("Dataset loading complete!")
print("Loading time: %s" % (time.time() - start_time))
