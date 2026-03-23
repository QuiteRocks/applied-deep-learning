from torch import nn
import torch

class ParityLSTM(nn.Module) : #Q2

    def __init__(self, hidden_dim=16):
        super().__init__()
        ###########################################
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
        ###########################################


    def forward(self, x, x_lens):
        ###########################################
        output, _ = self.lstm(x)
        idx = torch.arange(output.size(0), device=output.device)
        h_last = output[idx, x_lens -1]

        out = self.fc(h_last)
        return out
        #
        ###########################################
