###################################################################################################
#
# Copyright (C) 2024 Indoor Environment Dataset
#
###################################################################################################
"""
Indoor Environment CTF Dataset (K-Fold variants)
Classes: Classroom, Corridor, Lab, Staircase (4 classes)
Data format: (batch, 101, 2) - 101 frequency bins with real/imaginary components

Supports K-Fold splitting via environment variables:
  - INDOOR_KFOLD_K: number of folds (K). If K < 2, defaults to original 75/25 split
  - INDOOR_KFOLD_FOLD: which fold to use as test set [0..K-1]
  - INDOOR_KFOLD_REPEAT_SEED: optional integer to seed the fold permutation
  - INDOOR_KFOLD_MODE: 'sample' (default) for standard KFold over samples, or 'group' for
                       GroupKFold by grid point (leakage-free)

GroupKFold mode keeps all 200 measurements of a grid point together to avoid leakage.
"""
import os
import random
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision import transforms

import ai8x

from .indoor_environment import GlobalMinMaxNormalize


def _compute_kfold_indices(num_items: int, k_folds: int, fold_index: int, seed: int):
    """Deterministically compute train/test indices for K-Fold.

    - Shuffles 0..num_items-1 with the given seed
    - Splits into K contiguous folds (first `remainder` folds get one extra)
    - Returns (train_indices, test_indices) for the requested `fold_index`
    """
    assert 0 <= fold_index < k_folds
    indices = np.arange(num_items)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    base_fold_size = num_items // k_folds
    remainder = num_items % k_folds
    fold_sizes = [base_fold_size + (1 if i < remainder else 0) for i in range(k_folds)]

    start = sum(fold_sizes[:fold_index])
    end = start + fold_sizes[fold_index]
    test_indices = indices[start:end]
    # Use setdiff1d for deterministic order based on shuffled indices
    train_indices = np.setdiff1d(indices, test_indices, assume_unique=False)
    return train_indices, test_indices


class IndoorEnvironmentDataset(Dataset):
    """
    Indoor Environment CTF Dataset with GroupKFold capability via env vars
    """

    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform

        # Read KFold controls from environment (fallback to original split if not provided)
        self.k_folds = int(os.getenv('INDOOR_KFOLD_K', '0') or '0')
        self.fold_index = int(os.getenv('INDOOR_KFOLD_FOLD', '0') or '0')
        self.repeat_seed = int(os.getenv('INDOOR_KFOLD_REPEAT_SEED', '0') or '0')

        # Seed PRNGs for reproducibility across train/test instances
        base_seed = 42 + self.repeat_seed
        random.seed(base_seed)
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)

        # Load and preprocess data
        self._load_data()

    def _load_data(self):
        """Load .mat files and preprocess the CTF data"""
        data_files = {
            'CTF_class_mov': 'CTF_Class_mov_final.mat',
            'CTF_class_static': 'CTF_Class_static_final.mat',
            'CTF_corridor_mov': 'CTF_Corridor_mov_final.mat',
            'CTF_corridor_static': 'CTF_Corridor_static_final.mat',
            'CTF_lab_mov': 'CTF_Lab_mov_final.mat',
            'CTF_lab_static': 'CTF_Lab_static_final.mat',
            'CTF_SC_mov': 'CTF_SC_mov_final.mat',
            'CTF_SC_static': 'CTF_SC_static_final.mat'
        }

        # Load all .mat files
        data = {}
        for key, filename in data_files.items():
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
            data[key] = scipy.io.loadmat(filepath)

        # Extract arrays (4th item in each loaded variable)
        CTF_class_static_array = data['CTF_class_static'][list(data['CTF_class_static'].keys())[3]].T
        CTF_corridor_static_array = data['CTF_corridor_static'][list(data['CTF_corridor_static'].keys())[3]].T
        CTF_lab_static_array = data['CTF_lab_static'][list(data['CTF_lab_static'].keys())[3]].T
        CTF_SC_static_array = data['CTF_SC_static'][list(data['CTF_SC_static'].keys())[3]].T

        CTF_class_mov_array = data['CTF_class_mov'][list(data['CTF_class_mov'].keys())[3]].T
        CTF_corridor_mov_array = data['CTF_corridor_mov'][list(data['CTF_corridor_mov'].keys())[3]].T
        CTF_lab_mov_array = data['CTF_lab_mov'][list(data['CTF_lab_mov'].keys())[3]].T
        CTF_SC_mov_array = data['CTF_SC_mov'][list(data['CTF_SC_mov'].keys())[3]].T

        # Combine static and moving data for each environment
        CTF_class = np.concatenate((CTF_class_static_array, CTF_class_mov_array), axis=0)
        CTF_corridor = np.concatenate((CTF_corridor_static_array, CTF_corridor_mov_array), axis=0)
        CTF_lab = np.concatenate((CTF_lab_static_array, CTF_lab_mov_array), axis=0)
        CTF_SC = np.concatenate((CTF_SC_static_array, CTF_SC_mov_array), axis=0)

        num_grid_points = 194 * 2  # 388
        num_measurements = 200

        mode = os.getenv('INDOOR_KFOLD_MODE', 'sample').lower()
        use_kfold = self.k_folds and self.k_folds >= 2

        if mode == 'group':
            # GroupKFold-style (by grid point, leakage-free)
            if use_kfold:
                train_points, test_points = _compute_kfold_indices(
                    num_grid_points, self.k_folds, self.fold_index, seed=1234 + self.repeat_seed
                )
            else:
                num_train_points = int(0.75 * num_grid_points)  # 291
                rng = np.random.RandomState(1234 + self.repeat_seed)
                perm = rng.permutation(num_grid_points)
                train_points = perm[:num_train_points]
                test_points = perm[num_train_points:]

            train_set = []
            test_set = []
            for array in [CTF_class, CTF_corridor, CTF_lab, CTF_SC]:
                reshaped_array = array.reshape(num_grid_points, num_measurements, -1, 2)
                train_set.append(reshaped_array[train_points])
                test_set.append(reshaped_array[test_points])

            train_set = np.concatenate(train_set, axis=0)
            test_set = np.concatenate(test_set, axis=0)

            large_X_train = train_set.reshape([-1, 101, 2])
            large_X_test = test_set.reshape([-1, 101, 2])

            env_train_counts = [len(train_points) * num_measurements] * 4
            env_test_counts = [len(test_points) * num_measurements] * 4
            labels_train = [np.ones([env_train_counts[i]]) * i for i in range(4)]
            labels_test = [np.ones([env_test_counts[i]]) * i for i in range(4)]
            large_Y_train = np.concatenate(labels_train)
            large_Y_test = np.concatenate(labels_test)

        else:
            # Sample-level KFold (standard KFold) — may leak across measurements of same grid point
            # Build a single flattened dataset per environment first
            all_X = []
            all_Y = []
            for label, array in enumerate([CTF_class, CTF_corridor, CTF_lab, CTF_SC]):
                reshaped_array = array.reshape(num_grid_points, num_measurements, -1, 2)
                flat_array = reshaped_array.reshape(-1, 101, 2)
                all_X.append(flat_array)
                all_Y.append(np.ones([flat_array.shape[0]]) * label)

            large_X_all = np.concatenate(all_X, axis=0)
            large_Y_all = np.concatenate(all_Y, axis=0)

            total = large_X_all.shape[0]
            if use_kfold:
                train_idx, test_idx = _compute_kfold_indices(
                    total, self.k_folds, self.fold_index, seed=1234 + self.repeat_seed
                )
            else:
                # Fallback 75/25 sample-level split
                rng = np.random.RandomState(1234 + self.repeat_seed)
                perm = rng.permutation(total)
                split = int(0.75 * total)
                train_idx, test_idx = perm[:split], perm[split:]

            large_X_train = large_X_all[train_idx]
            large_Y_train = large_Y_all[train_idx]
            large_X_test = large_X_all[test_idx]
            large_Y_test = large_Y_all[test_idx]

        # Shuffle within each split (deterministically)
        rng_shuffle = np.random.RandomState(5678 + self.repeat_seed)
        shuffle_index_train = rng_shuffle.permutation(len(large_X_train))
        shuffle_index_test = rng_shuffle.permutation(len(large_X_test))

        large_X_train_shuffled = large_X_train[shuffle_index_train, :, :]
        large_Y_train_shuffled = large_Y_train[shuffle_index_train]

        large_X_test_shuffled = large_X_test[shuffle_index_test, :, :]
        large_Y_test_shuffled = large_Y_test[shuffle_index_test]

        if self.train:
            self.data = torch.tensor(large_X_train_shuffled, dtype=torch.float32)
            self.targets = torch.tensor(large_Y_train_shuffled, dtype=torch.long)
        else:
            self.data = torch.tensor(large_X_test_shuffled, dtype=torch.float32)
            self.targets = torch.tensor(large_Y_test_shuffled, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample, target = self.data[index], self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def indoor_environment_kfold_get_datasets(data, load_train=True, load_test=True):
    """Load Indoor Environment CTF dataset with GroupKFold split.

    Uses GlobalMinMaxNormalize + ai8x.normalize to match hardware dynamic range.
    No augmentations - uses same normalization as the base dataset.
    """
    (data_dir, args) = data

    # Use global min-max normalization (no augmentations)
    common_transform = transforms.Compose([
        GlobalMinMaxNormalize(),
        ai8x.normalize(args=args)
    ])

    if load_train:
        train_dataset = IndoorEnvironmentDataset(
            root=data_dir,
            train=True,
            transform=common_transform,
            download=False
        )
    else:
        train_dataset = None

    if load_test:
        test_dataset = IndoorEnvironmentDataset(
            root=data_dir,
            train=False,
            transform=common_transform,
            download=False
        )
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'IndoorEnvironmentKFold',
        'input': (1, 101, 2),
        'output': ('classroom', 'corridor', 'lab', 'sports-complex'),
        'loader': indoor_environment_kfold_get_datasets,
    },
]


