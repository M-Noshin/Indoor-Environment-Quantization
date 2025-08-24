###################################################################################################
#
# Copyright (C) 2024 Indoor Environment Dataset
#
###################################################################################################
"""
Indoor Environment CTF Dataset for Channel Transfer Function classification
Classes: Classroom, Corridor, Lab, Staircase (4 classes)
Data format: (batch, 101, 2) - 101 frequency bins with real/imaginary components
"""
import os
import sys
import random
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms

import ai8x


# ------------------------------------
# Transform Helper
# ------------------------------------


class GlobalMinMaxNormalize:
    """Scale a tensor to the [0, 1] interval using global dataset statistics.
    
    This preserves relative magnitude information between samples, unlike per-sample normalization.
    Based on analysis: dataset range is approximately [-0.011066, 0.011379]
    """
    
    def __init__(self, global_min=-0.011066, global_max=0.011379):
        self.global_min = global_min
        self.global_max = global_max
        self.global_range = global_max - global_min
    
    def __call__(self, tensor):
        return (tensor - self.global_min) / self.global_range



class IndoorEnvironmentDataset(Dataset):
    """
    Indoor Environment CTF Dataset
    Loads Channel Transfer Function data from .mat files and preprocesses for classification
    """
    
    def __init__(self, root, train=True, transform=None, download=False):
        """
        Args:
            root (string): Root directory of dataset
            train (bool): If True, creates dataset from training data
            transform (callable, optional): Optional transform to be applied on a sample
            download (bool): If true, downloads the dataset from the internet to root
        """
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Load and preprocess data
        self._load_data()
        
    def _load_data(self):
        """Load .mat files and preprocess the CTF data"""
        # Define paths to .mat files
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
        
        # Data splitting logic (exactly from the notebook)
        num_grid_points = 194 * 2  # 388
        num_measurements = 200
        num_train_points = int(0.75 * num_grid_points)  # 291
        
        train_set = []
        test_set = []
        
        # For each array (representing a different environment)
        for array in [CTF_class, CTF_corridor, CTF_lab, CTF_SC]:
            # Reshape the array to separate the measurements for each grid point
            reshaped_array = array.reshape(num_grid_points, num_measurements, -1, 2)
            
            # Randomly select grid points for training
            train_points = random.sample(range(num_grid_points), num_train_points)
            #print(len(train_points))
                
            # Get boolean array for test points
            test_points_bool = ~np.isin(range(num_grid_points), train_points)
            
            # Add the selected grid points to the training set and the rest to the test set
            train_set.append(reshaped_array[train_points])
            test_set.append(reshaped_array[test_points_bool])
            
            # ~ is the logical NOT operator, so ~np.isin(shuffle_index, train_points) gives all the indices not in train_points.
        # Concatenate the data from all environments
        train_set = np.concatenate(train_set, axis=0)
        test_set = np.concatenate(test_set, axis=0)

        
        # Reshape to final format
        large_X_train = train_set.reshape([-1, 101, 2])  # (232800, 101, 2)
        large_X_test = test_set.reshape([-1, 101, 2])    # (77600, 101, 2)
        
        # Create labels (4 classes: 0=Classroom, 1=Corridor, 2=Lab, 3=Staircase)
        # Training labels
        label1 = np.ones([291 * 200]) * 0  # classroom
        label2 = np.ones([291 * 200]) * 1  # corridor  
        label3 = np.ones([291 * 200]) * 2  # lab
        label4 = np.ones([291 * 200]) * 3  # staircase

        large_Y_train = np.concatenate([label1, label2, label3, label4])
        
        # Test labels
        label5 = np.ones([97 * 200]) * 0  # classroom
        label6 = np.ones([97 * 200]) * 1  # corridor
        label7 = np.ones([97 * 200]) * 2  # lab
        label8 = np.ones([97 * 200]) * 3  # staircase

        large_Y_test = np.concatenate([label5, label6, label7, label8])
        
        # Shuffle training data
        shuffle_index1 = random.sample(range(0,232800), 232800)
        # large_X_train_shuffled = large_X_train[shuffle_index1, :, :]
        large_X_train_new = large_X_train[shuffle_index1, :, :]
        # large_Y_train_shuffled = large_Y_train[shuffle_index1]
        large_Y_train = large_Y_train[shuffle_index1]

        
        # Shuffle test data
        shuffle_index2 = random.sample(range(0,77600), 77600)

        large_X_test_new = large_X_test[shuffle_index2, :, :]
        large_Y_test = large_Y_test[shuffle_index2]
        
        # Store appropriate split
        if self.train:
            self.data = torch.tensor(large_X_train_new, dtype=torch.float32)
            self.targets = torch.tensor(large_Y_train, dtype=torch.long)
        else:
            self.data = torch.tensor(large_X_test_new, dtype=torch.float32)
            self.targets = torch.tensor(large_Y_test, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class index
        """
        sample, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample, target


def indoor_environment_get_datasets(data, load_train=True, load_test=True):
    """
    Load Indoor Environment CTF dataset
    """
    (data_dir, args) = data

    # ------------------------------------------------------------------
    # NOTE:
    # All AI85/AI87 data loaders must normalize their inputs to match the
    # dynamic range expected by hardware simulation ( and eventually the
    # target device).  Failing to do so will typically lead to excellent
    # floating-point accuracy during training, but a dramatic drop once the
    # model is quantized or evaluated in simulated hardware – exactly what
    # we observed.
    #
    # The ai8x.normalize() transform converts an input in the [0, 1] range
    # to either:
    #   • floating-point [-128/128, +127/128] when args.act_mode_8bit == False
    #   • signed 8-bit integer [-128, +127]          when args.act_mode_8bit == True
    #
    # Our raw CTF tensors are converted to [0,1] range using global min-max
    # normalization, then ai8x.normalize() maps to hardware range.
    # ------------------------------------------------------------------

    # Global min-max normalization preserves relative magnitudes between samples
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
        'name': 'IndoorEnvironment',
        'input': (1, 101, 2), #(channels, height, width)
        'output': ('classroom', 'corridor', 'lab', 'sports-complex'),
        'loader': indoor_environment_get_datasets,
    },
] 