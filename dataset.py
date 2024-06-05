
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Callable


class VisualOdometryDataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        transform: Callable,
        sequence_length: int,
        validation: bool = False
    ) -> None:

        self.sequences = []

        directories = [d for d in os.listdir(
            dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        for subdir in directories:

            aux_path = f"{dataset_path}/{subdir}"

            # read data
            rgb_paths = self.read_images_paths(aux_path)

            if not validation:
                ground_truth_data = self.read_ground_truth(aux_path)
                interpolated_ground_truth = self.interpolate_ground_truth(
                    rgb_paths, ground_truth_data)
                
            # print(len(rgb_paths))
            # print(len(interpolated_ground_truth))
            # misma len

            # TODO: create sequences
            for i in range(1, len(rgb_paths), 2):

                if not validation:
                    position_first_image = np.array(interpolated_ground_truth[i-1][1])
                    position_second_image = np.array(interpolated_ground_truth[i][1])
                    difference = np.subtract(position_second_image, position_first_image)
                else:
                    difference = None

                self.sequences.append((rgb_paths[i-1][0], #timestamp imagen i-1
                                       rgb_paths[i-1][1], # path imagen i-1
                                       rgb_paths[i][0], # timestamp imagen i
                                       rgb_paths[i][1], # path imagen i
                                       difference)) # diferencia entre las posiciones

        self.transform = transform
        self.sequence_length = sequence_length
        self.validation = validation

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.TensorType:

        # Load sequence of images
        sequence_images = []
        # ground_truth_pos = []
        timestampt = 0

        _, path_i_minus_1, timestamp_i, path_i, difference = self.sequences[idx]

        # Load images
        image_i_minus_1 = cv2.imread(path_i_minus_1)
        image_i = cv2.imread(path_i)

        # Convert images to RGB (OpenCV loads images in BGR format)
        image_i_minus_1 = cv2.cvtColor(image_i_minus_1, cv2.COLOR_BGR2RGB)
        image_i = cv2.cvtColor(image_i, cv2.COLOR_BGR2RGB)

        # Apply transformations
        image_i_minus_1 = self.transform(image_i_minus_1)
        image_i = self.transform(image_i)

        sequence_images.append(image_i_minus_1)
        sequence_images.append(image_i)

        # ground_truth_pos.append(difference)
        timestampt = timestamp_i

        sequence_images = torch.stack(sequence_images)

        if difference is None:
            return sequence_images, torch.Tensor([0]), timestampt
        else:
            return sequence_images, torch.Tensor(difference), timestampt

    def read_images_paths(self, dataset_path: str) -> Tuple[float, str]:

        paths = []

        with open(f"{dataset_path}/rgb.txt", "r") as file:
            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                image_path = f"{dataset_path}/{line[1]}"

                paths.append((timestamp, image_path))

        return paths

    def read_ground_truth(self, dataset_path: str) -> Tuple[float, Tuple[float]]:

        ground_truth_data = []

        with open(f"{dataset_path}/groundtruth.txt", "r") as file:

            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                position = list(map(float, line[1:]))
                ground_truth_data.append((timestamp, position))

        return ground_truth_data

    def interpolate_ground_truth(
            self,
            rgb_paths: Tuple[float, str],
            ground_truth_data: Tuple[float, Tuple[float]]
    ) -> Tuple[float, Tuple[float]]:

        rgb_timestamps = [rgb_path[0] for rgb_path in rgb_paths]
        ground_truth_timestamps = [item[0] for item in ground_truth_data]

        # Interpolate ground truth positions for each RGB image timestamp
        interpolated_ground_truth = []

        for rgb_timestamp in rgb_timestamps:

            nearest_idx = np.argmin(
                np.abs(np.array(ground_truth_timestamps) - rgb_timestamp))

            interpolated_position = ground_truth_data[nearest_idx]
            interpolated_ground_truth.append(interpolated_position)

        return interpolated_ground_truth
