import torch
import torch.nn.functional as F
import torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1800):
        super().__init__()
        pe = torch.zeros(max_len, d_model)    # Shape: (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)      # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)      # Odd indices
        self.pe = pe.unsqueeze(1)                         # Shape: (max_len, 1, d_model)

    def forward(self, x):
        x = x + self.pe[:x.size(0)].to(x.device)
        return x

class TokenAlignedMaimaiAnalyzerMOdel(nn.Module):
    def __init__(self, input_dim=18, nhead=2, hidden_dim=128, num_layers=3, hidden_neuron=1, max_len=2200):
        super().__init__()
        self.hn = hidden_neuron
        self.pos_encoder = PositionalEncoding(input_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(input_dim, self.hn)
        if self.hn != 1:
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(self.hn, 1)
        self.activation = nn.Sigmoid()

    def forward(self, src):
        src = self.pos_encoder(src)       # Add positional encoding
        output = self.transformer_encoder(src)    # Transformer encoder
        output = output.mean(dim=0)       # Global average pooling
        output = self.decoder(output)     # Linear layer
        if self.hn != 1:
            output = F.relu(output)
            output = self.fc1(output)
        output = self.activation(output)

        return output