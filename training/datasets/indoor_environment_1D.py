###################################################################################################
#
# Copyright (C) 2024 Indoor Environment Dataset
#
###################################################################################################
"""
Indoor Environment CTF Dataset for Channel Transfer Function classification
Classes: Classroom, Corridor, Lab, Staircase (4 classes)
Raw data format per sample: [101, 2] -> 101 frequency bins × {real, imag}
Returned to model as: [2, 101] for Conv1d
"""
import os
import random
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision import transforms
import ai8x


# ------------------------------------
# Transform Helper
# ------------------------------------
class GlobalMinMaxNormalize:
    """Scale a tensor to [0, 1] using global dataset statistics.
    Preserves relative magnitude info between samples (vs per-sample scaling).
    Based on analysis: dataset range ≈ [-0.011066, 0.011379]
    """
    def __init__(self, global_min=-0.011066, global_max=0.011379):
        self.global_min = global_min
        self.global_max = global_max
        self.global_range = global_max - global_min

    def __call__(self, tensor):
        return (tensor - self.global_min) / self.global_range


class Slice1DLength:
    """Slice a [L, C] tensor to target length along the first dimension.
    If target_length >= current length, returns input unchanged.
    """
    def __init__(self, target_length: int, mode: str = 'center'):
        self.target_length = int(target_length)
        self.mode = mode

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.target_length is None:
            return tensor
        length = tensor.shape[0]
        if self.target_length >= length:
            return tensor
        if self.mode == 'center':
            start = max((length - self.target_length) // 2, 0)
        else:
            start = 0
        end = start + self.target_length
        return tensor[start:end, :]


class IndoorEnvironmentDataset(Dataset):
    """
    Indoor Environment CTF Dataset
    Loads Channel Transfer Function data from .mat files and preprocesses for classification.
    Returns tensors shaped [2, 101] (channels-first) for Conv1d models.
    """
    def __init__(self, root, train=True, transform=None, download=False):
        """
        Args:
            root (str): Root directory of dataset (.mat files)
            train (bool): If True, creates dataset from training split
            transform (callable, optional): Transform applied on each sample
            download (bool): Unused (kept for API compatibility)
        """
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform

        # Use the global random state set by the training framework (distiller.set_seed)
        # This ensures different seeds create different train/test splits
        # No need to override the seed here - it's already set by the training script

        self._load_data()

    def _load_data(self):
        """Load .mat files and preprocess the CTF data."""
        data_files = {
            'CTF_class_mov':     'CTF_Class_mov_final.mat',
            'CTF_class_static':  'CTF_Class_static_final.mat',
            'CTF_corridor_mov':  'CTF_Corridor_mov_final.mat',
            'CTF_corridor_static':'CTF_Corridor_static_final.mat',
            'CTF_lab_mov':       'CTF_Lab_mov_final.mat',
            'CTF_lab_static':    'CTF_Lab_static_final.mat',
            'CTF_SC_mov':        'CTF_SC_mov_final.mat',
            'CTF_SC_static':     'CTF_SC_static_final.mat',
        }

        data = {}
        for key, filename in data_files.items():
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
            data[key] = scipy.io.loadmat(filepath)

        # Extract arrays (4th item in each loaded variable)
        CTF_class_static_array    = data['CTF_class_static'][list(data['CTF_class_static'].keys())[3]].T
        CTF_corridor_static_array = data['CTF_corridor_static'][list(data['CTF_corridor_static'].keys())[3]].T
        CTF_lab_static_array      = data['CTF_lab_static'][list(data['CTF_lab_static'].keys())[3]].T
        CTF_SC_static_array       = data['CTF_SC_static'][list(data['CTF_SC_static'].keys())[3]].T

        CTF_class_mov_array       = data['CTF_class_mov'][list(data['CTF_class_mov'].keys())[3]].T
        CTF_corridor_mov_array    = data['CTF_corridor_mov'][list(data['CTF_corridor_mov'].keys())[3]].T
        CTF_lab_mov_array         = data['CTF_lab_mov'][list(data['CTF_lab_mov'].keys())[3]].T
        CTF_SC_mov_array          = data['CTF_SC_mov'][list(data['CTF_SC_mov'].keys())[3]].T

        # Combine static and moving data for each environment
        CTF_class    = np.concatenate((CTF_class_static_array,    CTF_class_mov_array),    axis=0)
        CTF_corridor = np.concatenate((CTF_corridor_static_array, CTF_corridor_mov_array), axis=0)
        CTF_lab      = np.concatenate((CTF_lab_static_array,      CTF_lab_mov_array),      axis=0)
        CTF_SC       = np.concatenate((CTF_SC_static_array,       CTF_SC_mov_array),       axis=0)

        # Spatial split (train/val/test) by grid points (keeps positions disjoint)
        num_grid_points = 194 * 2  # 388
        num_measurements = 200

        # Number of grid points to select for training/test
        num_train_points = int(0.75 * num_grid_points)  # 291
        num_test_points = num_grid_points - num_train_points  # 97

        # Fractions inside the training set for validation
        val_frac = 0.1
        num_val_points = int(val_frac * num_train_points)  # ~29
        num_final_train_points = num_train_points - num_val_points  # 262

        # Initialize empty lists
        train_set, val_set, test_set = [], [], []

        for array in [CTF_class, CTF_corridor, CTF_lab, CTF_SC]:
            # Separate measurements per grid point:
            # shape -> [grid_points, measurements, 101, 2]
            reshaped_array = array.reshape(num_grid_points, num_measurements, -1, 2)

            # Shuffle grid points
            grid_points = list(range(num_grid_points))
            random.shuffle(grid_points)

            # Split into train/test
            train_points = grid_points[:num_train_points]
            test_points = grid_points[num_train_points:]

            # Now split train into train/val
            val_points = train_points[:num_val_points]
            final_train_points = train_points[num_val_points:]

            # Verify spatial separation
            assert set(final_train_points).isdisjoint(set(val_points))
            assert set(final_train_points).isdisjoint(set(test_points))
            assert set(val_points).isdisjoint(set(test_points))

            # Add to each set
            train_set.append(reshaped_array[final_train_points])
            val_set.append(reshaped_array[val_points])
            test_set.append(reshaped_array[test_points])

        # Concatenate across environments
        train_set = np.concatenate(train_set, axis=0)
        val_set = np.concatenate(val_set, axis=0)
        test_set = np.concatenate(test_set, axis=0)

        # Final shape per sample: [101, 2]
        large_X_train = train_set.reshape([-1, 101, 2])  # ~209,600
        large_X_val = val_set.reshape([-1, 101, 2])      # ~23,200  
        large_X_test = test_set.reshape([-1, 101, 2])    # ~77,600

        # Create Labels for training data
        # Training labels (262 grid points per environment)
        label1 = np.zeros(num_final_train_points * 200, dtype=np.int64)  # classroom
        label2 = np.ones(num_final_train_points * 200, dtype=np.int64)   # corridor
        label3 = np.full(num_final_train_points * 200, 2, dtype=np.int64)  # lab
        label4 = np.full(num_final_train_points * 200, 3, dtype=np.int64)  # staircase
        large_Y_train = np.concatenate([label1, label2, label3, label4])

        # Validation labels (29 grid points per environment)
        label5 = np.zeros(num_val_points * 200, dtype=np.int64)  # classroom
        label6 = np.ones(num_val_points * 200, dtype=np.int64)   # corridor
        label7 = np.full(num_val_points * 200, 2, dtype=np.int64)  # lab
        label8 = np.full(num_val_points * 200, 3, dtype=np.int64)  # staircase
        large_Y_val = np.concatenate([label5, label6, label7, label8])

        # Test labels (97 grid points per environment)
        label9 = np.zeros(num_test_points * 200, dtype=np.int64)   # classroom
        label10 = np.ones(num_test_points * 200, dtype=np.int64)   # corridor
        label11 = np.full(num_test_points * 200, 2, dtype=np.int64)  # lab
        label12 = np.full(num_test_points * 200, 3, dtype=np.int64)  # staircase
        large_Y_test = np.concatenate([label9, label10, label11, label12])

        # Shuffle within each split (but maintain spatial separation)
        shuffle_index1 = random.sample(range(len(large_X_train)), len(large_X_train))
        large_X_train = large_X_train[shuffle_index1, :, :]
        large_Y_train = large_Y_train[shuffle_index1]

        shuffle_index2 = random.sample(range(len(large_X_val)), len(large_X_val))
        large_X_val = large_X_val[shuffle_index2, :, :]
        large_Y_val = large_Y_val[shuffle_index2]

        shuffle_index3 = random.sample(range(len(large_X_test)), len(large_X_test))
        large_X_test = large_X_test[shuffle_index3, :, :]
        large_Y_test = large_Y_test[shuffle_index3]

        # Store split as tensors (still [101, 2]; we'll permute to [2, 101] in __getitem__)
        if self.train:
            # Combine train and validation data for ai8x framework compatibility
            # The framework will use indices to separate them
            combined_X = np.concatenate([large_X_train, large_X_val], axis=0)
            combined_Y = np.concatenate([large_Y_train, large_Y_val], axis=0)
            
            self.data = torch.tensor(combined_X, dtype=torch.float32)
            self.targets = torch.tensor(combined_Y, dtype=torch.long)
            
            # Create valid_indices for ai8x framework
            # These point to the validation portion of the combined dataset
            self.valid_indices = list(range(len(large_X_train), len(large_X_train) + len(large_X_val)))
        else:
            self.data = torch.tensor(large_X_test, dtype=torch.float32)
            self.targets = torch.tensor(large_Y_test, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns:
            sample: torch.FloatTensor [2, 101]  (channels-first for Conv1d)
            target: torch.LongTensor scalar in {0..3}
        """
        sample, target = self.data[index], self.targets[index]  # sample: [101, 2]

        # Apply transforms on [101, 2] (min-max -> [0,1], then ai8x.normalize)
        if self.transform is not None:
            sample = self.transform(sample)

        # Convert to channels-first for Conv1d: [2, 101]
        sample = sample.permute(1, 0).contiguous()
        return sample, target


def indoor_environment_get_datasets(data, load_train=True, load_test=True):
    """
    Load Indoor Environment CTF dataset for ai8x-training.
    Returns (train_dataset, test_dataset).
    """
    (data_dir, args) = data

    # Optional 1D length override via --input-1d-length
    tfs = []
    if hasattr(args, 'input_1d_length') and args.input_1d_length is not None:
        if int(args.input_1d_length) > 0 and int(args.input_1d_length) < 101:
            tfs.append(Slice1DLength(int(args.input_1d_length), mode='center'))
    # Map raw -> [0,1] (global min-max), then -> hw range via ai8x.normalize()
    tfs.extend([GlobalMinMaxNormalize(), ai8x.normalize(args=args)])
    common_transform = transforms.Compose(tfs)

    train_dataset = IndoorEnvironmentDataset(
        root=data_dir, train=True,  transform=common_transform
    ) if load_train else None

    test_dataset = IndoorEnvironmentDataset(
        root=data_dir, train=False, transform=common_transform
    ) if load_test else None

    return train_dataset, test_dataset


datasets = [
    {
        'name':  'IndoorEnvironment_1D',
        'input': (2, 101),  # (channels, length) for Conv1d models
        'output': ('classroom', 'corridor', 'lab', 'staircase'),
        'loader': indoor_environment_get_datasets,
    },
]
