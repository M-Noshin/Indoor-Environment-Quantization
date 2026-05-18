###################################################################################################
#
# Copyright (C) 2024 Indoor Environment Classification Model
#
###################################################################################################
"""
Indoor Environment Classification network for AI85/AI87 (MAX78000/MAX78002)
Classifies indoor environments: Classroom, Corridor, Lab, Sports-Complex
"""
from torch import nn

import ai8x


class AI85IndoorEnvNetv2(nn.Module):
    """
    Indoor Environment Classification CNN for Channel Transfer Function (CTF) data
    Input: (batch, 101, 2) - 101 frequency bins with real/imaginary components
    Output: 4 classes (Classroom, Corridor, Lab, Sports-Complex)
    """

    def __init__(
            self,
            num_classes=4,
            num_channels=2,
            dimensions=(101, 1),  # pylint: disable=unused-argument
            bias=True, # Set to True by default for ai8x
            p_dropout=0.4,
            batchnorm="Affine",
            **kwargs
    ):
        super().__init__()
        
        # Determine input length from provided dimensions argument
        # dimensions is expected to be (length, 1) for 1D
        input_length = dimensions[0] if isinstance(dimensions, tuple) and len(dimensions) > 0 else 101
        
        # First Conv1D layer (optionally with BatchNorm) + ReLU
        conv1_cls = ai8x.FusedConv1dBNReLU if batchnorm in ("Affine", "NoAffine") else ai8x.FusedConv1dReLU
        self.conv1 = conv1_cls(
            in_channels=num_channels, 
            out_channels=10, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            bias=bias,
            batchnorm=batchnorm if conv1_cls is ai8x.FusedConv1dBNReLU else None,
            **kwargs
        )
        
        # Second Conv1D layer (optionally with BatchNorm) + ReLU
        conv2_cls = ai8x.FusedConv1dBNReLU if batchnorm in ("Affine", "NoAffine") else ai8x.FusedConv1dReLU
        self.conv2 = conv2_cls(
            in_channels=10, 
            out_channels=10, 
            kernel_size=3,
            stride=1, 
            padding=1,
            bias=bias,
            batchnorm=batchnorm if conv2_cls is ai8x.FusedConv1dBNReLU else None,
            **kwargs
        )
        
        # First fully connected layer with ReLU: 10 * input_length -> 200
        # FC layers: keep bias for learnable offsets
        self.fc1 = ai8x.FusedLinearReLU(
            in_features=10 * input_length, #1010 
            out_features=200,
            bias=bias,    # always on here
            **kwargs
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=p_dropout)
        
        # Final fully connected layer: 200 -> 4 classes
        self.fc2 = ai8x.Linear(
            in_features=200, 
            out_features=num_classes,
            bias=bias,  # always on here
            **kwargs
        )
        
    def forward(self, x):  # x: (B, 2, 101)
        x = self.conv1(x)          # -> (B, 10, 101)
        x = self.conv2(x)          # -> (B, 10, 101)
        x = x.flatten(1)           # -> (B, 10*101)
        x = self.fc1(x)            # -> (B, 200)
        x = self.dropout(x)        # -> (B, 200) with dropout
        x = self.fc2(x)            # -> (B, 4)  (logits)
        return x


def ai85indoorenvnetv2(pretrained=False, **kwargs):
    """
    Constructs a AI85IndoorEnvNetv2 model.
    """
    assert not pretrained
    return AI85IndoorEnvNetv2(**kwargs)


def ai85indoorenvnetv2_nobn(pretrained=False, **kwargs):
    """
    BatchNorm-free variant used for PTQ evaluation after running batchnormfuser.py.
    """
    assert not pretrained
    return AI85IndoorEnvNetv2(batchnorm=None, **kwargs)


models = [
    {
        'name': 'ai85indoorenvnetv2',
        'min_input': 1,
        'dim': 1,
    },
    {
        'name': 'ai85indoorenvnetv2_nobn',
        'min_input': 1,
        'dim': 1,
    },
] 