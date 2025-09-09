"""
GRU-based models for emotion recognition.

This module contains GRU-based models for emotion recognition tasks,
including a base GRU model and a specialized VAD regression model.

Example :
    >>> model = VADRegressionModel(input_size=768, hidden_size=256, num_layers=2)
    >>> outputs = model(features)
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2023-11-15"

import torch
import torch.nn as nn

# Ensure torch is used explicitly to satisfy linters
TORCH_VERSION = torch.__version__


class GRUModel(nn.Module):
    """
    Base GRU model with sequential layers.

    This is the base GRU model with a sequential architecture.

    Attributes:
        hidden_size (int): Size of the hidden layers
        num_layers (int): Number of GRU layers
        gru_model (nn.Sequential): Sequential container of GRU layers and other components

    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize the GRU model.

        Args :
            input_size (int): Dimension of input features
            hidden_size (int): Hidden size of GRU layers
            num_layers (int): Number of GRU layers
            output_size (int): Output dimension
            dropout (float): Dropout rate
        """
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_model = nn.Sequential(
            nn.GRU(
                input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
            ),
            nn.Linear(hidden_size, output_size),
            nn.GRU(
                hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout
            ),
            nn.Linear(hidden_size, output_size),
            nn.Embedding(output_size, output_size),
            nn.Linear(output_size, output_size),
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args :
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_size]

        Returns :
            torch.Tensor: Output tensor
        """
        out = self.gru_model(x)
        return out


class VADRegressionModel(nn.Module):
    """
    Wrapper for GRUModel to handle VAD regression.

    This model adapts the GRUModel for the specific task of VAD regression.
    It handles the architecture issues in the original GRUModel.

    Attributes :
        model (GRUModel): The underlying GRU model
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=3, dropout=0.2):
        """
        Initialize the VAD regression model.

        Args :
            input_size (int): Dimension of input features
            hidden_size (int): Hidden size of GRU layers
            num_layers (int): Number of GRU layers
            output_size (int): Output dimension (default: 3 for V, A, D)
            dropout (float): Dropout rate
        """
        super(VADRegressionModel, self).__init__()
        self.model = GRUModel(input_size, hidden_size, num_layers, output_size, dropout)

    def forward(self, x):
        """
        Forward pass through the model.

        Args :
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_size]

        Returns :
            torch.Tensor: Output tensor of shape [batch_size, output_size]
        """
        # The GRUModel has issues with its architecture, so we need to handle it properly
        # First GRU layer
        x, _ = self.model.gru_model[0](x)
        # Get the last time step output
        x = x[:, -1, :]
        # First linear layer
        x = self.model.gru_model[1](x)
        # We'll skip the rest of the model as it's not properly designed
        # Instead, we'll return the output directly as VAD predictions
        return x


class ImprovedVADModel(nn.Module):
    """
    Improved VAD regression model with proper GRU architecture.

    This model uses a more straightforward architecture for VAD regression,
    avoiding the issues in the original GRUModel.

    Attributes :
        gru (nn.GRU): GRU layer
        fc (nn.Linear): Fully connected output layer
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=3, dropout=0.2):
        """
        Initialize the improved VAD regression model.

        Args :
            input_size (int): Dimension of input features
            hidden_size (int): Hidden size of GRU layers
            num_layers (int): Number of GRU layers
            output_size (int): Output dimension (default: 3 for V, A, D)
            dropout (float): Dropout rate
        """
        super(ImprovedVADModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the model.

        Args :
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_size]

        Returns :
            torch.Tensor: Output tensor of shape [batch_size, output_size]
        """
        # GRU returns output and hidden state, we only need the output
        output, _ = self.gru(x)
        # Use the last time step output
        last_hidden = output[:, -1, :]
        # Project to output dimension
        vad_prediction = self.fc(last_hidden)
        return vad_prediction
