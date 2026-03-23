import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module): #Q1
    def __init__(self, in_channels):
      super().__init__()
      self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: Tensor) -> Tensor:
      x = x.permute(0, 2, 3, 1)
      x = self.norm(x)
      x = x.permute(0, 3, 1,2)
      return x

class ConvNextStem(nn.Module): #Q2
   def __init__(self, in_channels, out_channels, kernel_size=3):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)
    self.norm = LayerNorm2d(out_channels)

   def forward(self,x):
    x = self.conv(x)
    x = self.norm(x)
    return x

class ConvNextBlock(nn.Module): #Q3

  def __init__(self, d_in, layer_scale=1e-6, kernel_size=7, stochastic_depth_prob=1):
    super().__init__()

    self.sd_prob = stochastic_depth_prob
    pad = (kernel_size -1) //2
    self.dwconv = nn.Conv2d(d_in, d_in, kernel_size=kernel_size, padding = pad, groups=d_in)

    self.norm = LayerNorm2d(d_in)
    self.pwconv1 = nn.Conv2d(d_in, 4 * d_in, kernel_size=1)
    self.act = nn.GELU()
    self.pwconv2 = nn.Conv2d(4* d_in, d_in, kernel_size=1)

    self.gamma = nn.Parameter(torch.ones(d_in) * layer_scale)

  def forward(self,x):
    residual = x

    out = self.dwconv(x)
    out = self.norm(out)
    out = self.pwconv1(out)
    out = self.act(out)
    out = self.pwconv2(out)

    out = self.gamma.view(1, -1, 1, 1) * out

    if self.training:
      if self.sd_prob < 1.0 and torch.rand(1).item() > self.sd_prob:
        return residual
      return residual + out
    else:
      return residual + self.sd_prob * out

class ConvNextDownsample(nn.Module): #Q4
  def __init__(self, d_in, d_out, width=2):
    super().__init__()
    self.norm = LayerNorm2d(d_in) 
    self.conv = nn.Conv2d(d_in, d_out, kernel_size=width, stride=width) # using stride=width to reduce spatial dims

  def forward(self,x):
    x = self.norm(x)
    x = self.conv(x)
    return x

class ConvNextClassifier(nn.Module): #Q5
  def __init__(self, d_in, d_out):
    super().__init__()
    self.pool = nn.AdaptiveAvgPool2d(1) 
    self.flatten = nn.Flatten()
    self.norm = nn.LayerNorm(d_in)
    self.fc = nn.Linear(d_in, d_out)

  def forward(self,x):
    x = self.pool(x)
    x = self.flatten(x)
    x = self.norm(x)
    x = self.fc(x)
    return x 

class ConvNext(nn.Module): #Q6

  def __init__(self, in_channels, out_channels, blocks=[96]):
    super().__init__()

    L = len(blocks) #total number of residual blocks
    layers = []

    layers.append(ConvNextStem(in_channels, blocks[0]))

    block_idx = 0
    for i in range(len(blocks)):
      if i > 0 and blocks[i] != blocks[i-1]:
       layers.append(ConvNextDownsample(blocks[i-1], blocks[i]))

      sd_prob = 1.0 - (block_idx / L) * 0.5
      layers.append(ConvNextBlock(blocks[i], stochastic_depth_prob=sd_prob))
      block_idx += 1

    layers.append(ConvNextClassifier(blocks[-1], out_channels))
    self.network = nn.Sequential(*layers)

    # Q7
    for m in self.modules():
      if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
        if m.bias is not None:
          nn.init.zeros_(m.bias)
      elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
      
  def forward(self,x):
    return self.network(x)