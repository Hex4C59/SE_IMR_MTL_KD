import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_model = nn.Sequential(
            nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout),
            nn.Linear(hidden_size, output_size),
            nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout),
            nn.Linear(hidden_size, output_size),
            nn.Embedding(output_size, output_size),
            nn.Linear(output_size, output_size)
        )
        
    def forward(self, x):
        out = self.gru_model(x)
        return out