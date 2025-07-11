###################################################################################################
#
# Copyright (C) 2024 Indoor Environment Classification Model
#
###################################################################################################
"""
Indoor Environment Classification network for AI85/AI87 (MAX78000/MAX78002)
Classifies indoor environments: Classroom, Corridor, Lab, Staircase
"""
from torch import nn

import ai8x


class AI85IndoorEnvNetv1(nn.Module):
    """
    Indoor Environment Classification CNN for Channel Transfer Function (CTF) data
    Input: (batch, 101, 2) - 101 frequency bins with real/imaginary components
    Output: 4 classes (Classroom, Corridor, Lab, Staircase)
    """

    def __init__(
            self,
            num_classes=4,
            num_channels=1,
            dimensions=(101, 2),  # pylint: disable=unused-argument
            bias=False, # default off for conv layers
            **kwargs
    ):
        super().__init__()
        
        # First Conv2D layer with BatchNorm and ReLU fused: 5 filters, kernel (3,3)
        self.conv1 = ai8x.FusedConv2dBNReLU(
            in_channels=num_channels, 
            out_channels=5, 
            kernel_size=(3, 3), 
            stride=1, 
            padding=1,
            bias=bias,
            **kwargs
        )
        
        # Second Conv2D layer with BatchNorm and ReLU fused: 5 filters, kernel (3,3)
        self.conv2 = ai8x.FusedConv2dBNReLU(
            in_channels=5, 
            out_channels=5, 
            kernel_size=(3, 3), 
            stride=1, 
            padding=1,
            bias=bias,
            **kwargs
        )
        
        # First fully connected layer with ReLU: 5 * 101 * 2 -> 50
        # FC layers: keep bias for learnable offsets
        self.fc1 = ai8x.FusedLinearReLU(
            in_features=5 * 101 * 2,
            out_features=50,
            bias=True,    # always on here
            **kwargs
        )
        
        # Final fully connected layer: 50 -> 4 classes
        self.fc2 = ai8x.Linear(
            in_features=50, 
            out_features=num_classes,
            bias=True,  # always on here
            **kwargs
        )
        
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Reshape input from (batch, 101, 2) to (batch, 1, 101, 2)
        x = x.unsqueeze(1)
        
        # First conv layer
        x = self.conv1(x)
        
        # Second conv layer
        x = self.conv2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # First fully connected layer with ReLU
        x = self.fc1(x)
        
        # Final fully connected layer
        x = self.fc2(x)
        
        return x


def ai85indoorenvnetv1(pretrained=False, **kwargs):
    """
    Constructs a AI85IndoorEnvNetv1 model.
    """
    assert not pretrained
    return AI85IndoorEnvNetv1(**kwargs)


models = [
    {
        'name': 'ai85indoorenvnetv1',
        'min_input': 1,
        'dim': 2,
    },
] 