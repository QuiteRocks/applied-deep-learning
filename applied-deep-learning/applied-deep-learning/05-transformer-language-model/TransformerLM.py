import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x): #Q1
        B, L, d = x.shape

        t = torch.arange(L, dtype=torch.float32, device=x.device)
        j = torch.arange(0, d, 2, dtype=torch.float32, device=x.device)
        freqs = torch.exp(-math.log(10000.0) * j / d)
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)

        pe = torch.zeros(L, d, device=x.device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        return x + pe.unsqueeze(0)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=True)
        self.classifier = nn.Linear(d_model, vocab_size)

    def generateCausalMask(self, L): #Q2
        mask = torch.triu(torch.full((L, L), float('-inf')), diagonal=1)
        return mask

    def forward(self, x): #Q3
        e = self.embeddings(x)
        e = self.position(e)

        mask = self.generateCausalMask (x.shape[1]).to(x.device)
        h = self.encoder(e, mask=mask, is_causal=True)

        return self.classifier(h)