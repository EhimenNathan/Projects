import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FourierLayer, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize weights using Fourier features
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        # Fourier transform operation for the layer
        x_ft = torch.fft.fft2(x)
        x_ft = torch.fft.ifft2(x_ft * self.weights)
        return x_ft

class FNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FNOBlock, self).__init__()
        self.fourier_layer = FourierLayer(in_channels, out_channels, kernel_size)
        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.fourier_layer(x)
        x = self.fc(x)
        return x

class NestedFNO(nn.Module):
    def __init__(self, input_size, levels, num_channels):
        super(NestedFNO, self).__init__()
        self.levels = levels
        self.fno_blocks = nn.ModuleList([FNOBlock(input_size, num_channels, kernel_size=3) for _ in range(levels)])

    def forward(self, x):
        outputs = []
        for i in range(self.levels):
            x = self.fno_blocks[i](x)
            outputs.append(x)
        return outputs

# Example usage:
input_size = 64
levels = 5
num_channels = 32
model = NestedFNO(input_size, levels, num_channels)

# Random input tensor (simulating permeability, pressure, etc.)
x = torch.randn((1, 1, input_size, input_size))  # Batch size of 1, 1 channel
output = model(x)

# Visualizing output (plotting first level)
import matplotlib.pyplot as plt
plt.imshow(output[0].detach().numpy()[0, 0], cmap='viridis')
plt.colorbar()
plt.title("Predicted Pressure Build-Up at Level 0")
plt.show()

class OptimizedFNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(OptimizedFNOBlock, self).__init__()
        self.fourier_layer = FourierLayer(in_channels, out_channels, kernel_size)
        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.fourier_layer(x)
        x = F.relu(self.fc(x))  # Add ReLU activation for non-linearity
        return x

class OptimizedNestedFNO(NestedFNO):
    def __init__(self, input_size, levels, num_channels):
        super(OptimizedNestedFNO, self).__init__(input_size, levels, num_channels)

    def forward(self, x):
        x = x.to('cuda')  # Move data to GPU for faster computation
        outputs = []
        for i in range(self.levels):
            x = self.fno_blocks[i](x)
            outputs.append(x)
        return outputs