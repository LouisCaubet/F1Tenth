import os
import pickle
import time

from config_loader import *
from lidar_preprocessing import convert_lidar_to_image, restrict_lidar_fov


class DatasetGenerator:
    """Generates a dataset ready for training from the wall_follow.py exported data."""

    def __init__(self, name):

        path = os.path.join(DATASET_FOLDER, name + ".pickle")
        self.export_dir = os.path.join(DATASET_FOLDER, name)

        if os.path.isdir(self.export_dir):
            print("Dataset " + name + " already generated!")
            return

        os.mkdir(self.export_dir)
        os.mkdir(os.path.join(self.export_dir, "inputs"))

        with open(path, "rb") as file:
            self.dataset = pickle.load(file)

        self.targets = dict()

        for i, entry in enumerate(self.dataset):
            self.preprocessor(entry, i)

        with open(os.path.join(self.export_dir, "targets.pickle"), "wb+") as target_file:
            pickle.dump(self.targets, target_file)

    def preprocessor(self, entry: str, index: int) -> dict:
        """Preprocesses the raw data from the txt files. Creates an image from the lidar data
        and writes it to the input directory. Adds the target to the targets dictionnary. """

        input = restrict_lidar_fov(entry[0])
        output = [entry[1][0], entry[1][1]]

        img = convert_lidar_to_image(input)

        image_path = os.path.join(self.export_dir, "inputs", "input_" + str(index) + ".png")

        img.save(image_path, "PNG")
        img.close()

        self.targets[image_path] = output


print("Starting generation of datasets...")
start = time.time()

for i, x in enumerate(DATASET_LIST):
    DatasetGenerator(x)
    print("Dataset " + str(i) + "/" + str(len(DATASET_LIST)) + " generated...")
    if DEBUG_GENERATE_ONLY_ONE_DATASET:
        print("DEBUG_GENERATE_ONLY_ONE_DATASET enabled. Ignoring other datasets to generate.")
        break

print("Dataset generation complete (", time.time() - start, "s )")
