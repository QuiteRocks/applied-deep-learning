import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class PoSGRU(nn.Module) :# Q4 

    def __init__(self, vocab_size=1000, embed_dim=16, hidden_dim=16, num_layers=2, output_dim=10, residual=True) :
      super().__init__()
      ###########################################
      #
      self.residual = residual
      self.num_layers = num_layers

      self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
      self.proj = nn.Linear(embed_dim, hidden_dim)

      self.grus = nn.ModuleList()
      for i in range(num_layers):
        self.grus.append(nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True))

      self.classifier = nn.Sequential( 
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim)
      )
      #
      ###########################################
  

    

    def forward(self, x):# Q4
      ###########################################
      #
      h = self.embedding(x) # emebed word id into vectors 
      h = self.proj(h)
      for gru in self.grus:
        out, _ = gru(h)
        if self.residual:
          h = out + h
        else:
          h = out 

      logits = self.classifier(h) # classify each token 
      return logits
      #
      ###########################################