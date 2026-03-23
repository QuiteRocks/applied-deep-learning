import torch
torch.manual_seed(42)
import torchvision
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import datetime
import random
import string
import wandb 

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

config = {
    "bs":2048,   # batch size
    "lr":0.003, # learning rate 
    "l2reg":0.00005, # weight decay
    "lr_decay":0.99, # exponential learning decay
    "aug":True, # enable data augmentation
    "max_epoch":20
}


def main():

  # Get dataloaders
  train_loader, test_loader = getDataloaders(visualize=True)

  # Build model
  model = SimpleCNN()


  ###################################
  # Q1 Sanity Check
  ###################################
  x,y = next(iter(train_loader))
  out = model(x)
  assert(out.shape == (config["bs"], 10))
  
  # Start model training
  train(model, train_loader, test_loader)


###################################
# Q1 Data Augmentation
###################################
def getDataloaders(visualize = True):
  
  # Set up dataset and data loaders
  test_transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])

  if config["aug"]:
    train_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=10 , translate=(0.1, 0.1), scale=(0.9, 1.1)),   
        transforms.Normalize((0.1307,), (0.3081,))
        ])
  else:
    train_transform = test_transform



  train_set = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
  test_set = datasets.MNIST('../data', train=False, download=True, transform=test_transform)

  train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=config["bs"])
  test_loader = torch.utils.data.DataLoader(test_set,  shuffle=False, batch_size=config["bs"])

  if visualize:
    # Plot out some transforms
    to_pil_image = transforms.ToPILImage()
    img = to_pil_image(train_set.data[0])

    fig, axs = plt.subplots(3,10, figsize=(10, 3))
    axs[0][0].imshow(img, cmap="grey")
    axs[0][0].get_xaxis().set_visible(False)
    axs[0][0].get_yaxis().set_visible(False)
    for i in range(1,30):
      axs[i//10][i%10].imshow(train_transform(img).squeeze(), cmap="grey")
      axs[i//10][i%10].get_xaxis().set_visible(False)
      axs[i//10][i%10].get_yaxis().set_visible(False)
    plt.show()

  return train_loader, test_loader


###################################
# Q2 Implement SimpleCNN
###################################

class SimpleCNN(nn.Module):

  def __init__(self):
    super(SimpleCNN, self).__init__()

    self.conv1 = nn.Conv2d(1, 36, kernel_size=5, stride=1) #block 1
    self.conv2 = nn.Conv2d(36, 36, kernel_size=5, stride=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2)

    self.conv3 = nn.Conv2d(36, 64, kernel_size=3, stride=1) #block 2
    self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2)

    self.conv5 = nn.Conv2d(128, 256, kernel_size=1, stride=1) # blcok 3
    self.conv6 = nn.Conv2d(256, 10, kernel_size=1, stride=1)
    self.pool3 = nn.AdaptiveAvgPool2d(1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool1(x)

    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.pool2(x)

    x = F.relu(self.conv5(x))
    x = self.conv6(x)

    x = self.pool3(x)
    x = x.squeeze()
    return x

###################################
# Q3 Compute Accuracy
###################################

def computeAccuracy(out, y):
  _, pred = torch.max(out, dim=1)
  acc = (pred == y).float().mean().item()
  return acc
############################################
# Q4 Integrate wandb Logging
# -- login
# -- init
# -- per-batch logging of train acc & loss
# -- per epoch logging of test acc & loss
############################################

def train(model, train_loader, test_loader):

  #wandb.login(key="---")
  wandb.init( #wandb set up
    project="Minst CS435 A3",
    name= generateRunName(),
    config=config
  )  

  model.to(device)

  optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
  scheduler = ExponentialLR(optimizer, gamma=config["lr_decay"])

  criterion = nn.CrossEntropyLoss()

  iteration = 0
  for epoch in range(config["max_epoch"]):
    model.train()
   
    for x,y in train_loader:
      x = x.to(device)
      y = y.to(device)

      out = model(x)
      loss = criterion(out,y)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      acc = computeAccuracy(out, y)
      
      wandb.log({"Loss/train": loss.item(), "Acc/train": acc}, step=iteration)

      iteration+=1

    # Evaluate on held out data
    test_loss, test_acc = evaluate(model, test_loader)
    wandb.log({"Loss/test": test_loss, "Acc/test": test_acc}, step=iteration)
    scheduler.step()
  wandb.finish()



############################################
# Skeleton Code
############################################

def evaluate(model, test_loader):
  model.eval()

  running_loss = 0
  running_acc = 0
  criterion = torch.nn.CrossEntropyLoss(reduction="sum")

  for x,y in test_loader:

    x = x.to(device)
    y = y.to(device)

    out = model(x)
    loss = criterion(out,y)

    acc = computeAccuracy(out, y)*x.shape[0]

    running_loss += loss.item()
    running_acc += acc

  return running_loss/len(test_loader.dataset), running_acc/len(test_loader.dataset)
  

def generateRunName():
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  now = datetime.datetime.now()
  run_name = "["+random_string+"] MNIST  "+now.strftime("[%m-%d-%Y--%H:%M]")
  return run_name


def generatePredictionPlot(model, test_loader):
  model.eval()
  x,y = next(iter(test_loader))
  out = F.softmax(model(x.to(device)).detach(), dim=1)

  num = min(20, x.shape[0])
  f, axs = plt.subplots(2, num, figsize=(4*num,8))
  for i in range(0,num):
    axs[0,i].imshow(x[i,:,:].squeeze().cpu(), cmap='gray')
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)
    axs[1,i].bar(list(range(10)),out[i,:].squeeze().cpu(), label=list(range(10)))
    axs[1,i].set_xticks(list(range(10)))
    axs[1,i].set_ylim(0,1)

  return f


if __name__ == "__main__":
  main()