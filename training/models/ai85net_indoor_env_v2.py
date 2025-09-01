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
            **kwargs
    ):
        super().__init__()
        
        # First Conv2D layer with BatchNorm and ReLU fused: 5 filters, kernel (3,3)
        self.conv1 = ai8x.FusedConv1dBNReLU(
            in_channels=num_channels, 
            out_channels=10, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            bias=bias,
            batchnorm="Affine",
            **kwargs
        )
        
        # Second Conv2D layer with BatchNorm and ReLU fused: 5 filters, kernel (3,3)
        self.conv2 = ai8x.FusedConv1dBNReLU(
            in_channels=10, 
            out_channels=10, 
            kernel_size=3,
            stride=1, 
            padding=1,
            bias=bias,
            batchnorm="Affine",
            **kwargs
        )
        
        # First fully connected layer with ReLU: 5 * 101 -> 50
        # FC layers: keep bias for learnable offsets
        self.fc1 = ai8x.FusedLinearReLU(
            in_features=10 * 101, #505
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


models = [
    {
        'name': 'ai85indoorenvnetv2',
        'min_input': 1,
        'dim': 1,
    },
] 