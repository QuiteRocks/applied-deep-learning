import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import matplotlib.pyplot as plt
import datetime
import random
import string
import wandb
from tqdm import tqdm

# Import our own files
from data.cifar10 import getCIFAR10Dataloaders
from models.convnext import ConvNext

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"

config = {
    "bs":128,   # batch size
    "lr":0.004, # learning rate
    "l2reg":0.0000001, # weight decay
    "max_epoch":200,
    "blocks":[64,64,128,128,256,256,512,512]
}


def main():

  # Get dataloaders
  train_loader, val_loader, test_loader = getCIFAR10Dataloaders(config)

  # Build model
  model = ConvNext(3, 10, config["blocks"])
  print(model)

  torch.compile(model)


  # Start model training
  train(model, train_loader, val_loader)


############################################
# Skeleton Code
############################################

def computeAccuracy(out, y):
  _, predicted = torch.max(out.data, 1)
  acc = (predicted == y).sum().item()/out.shape[0]
  return acc

def train(model, train_loader, val_loader):

  # Log our exact model architecture string
  config["arch"] = str(model)
  run_name = generateRunName()

  # Startup wandb logging
  wandb.login()
  wandb.init(project="CIFAR10 CS499 A4", name=run_name, config=config)

  # Move model to the GPU
  model.to(device)

  # Set up optimizer and our learning rate schedulers
  optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])

  ################################
  #scheduler = Q8 warmup for 20 epochs then cosine decay
  ################################
  warmup_epochs = 20
  warmup_sched = LinearLR(optimizer, start_factor=0.25, total_iters=warmup_epochs)
  cosine_sched = CosineAnnealingLR(optimizer, T_max=config["max_epoch"] - warmup_epochs)
  scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
  # Set up our cross entropy loss
  criterion = nn.CrossEntropyLoss()

  # Main training loop with progress bar
  iteration = 0
  best_val = 0
  pbar = tqdm(total=config["max_epoch"]*len(train_loader), desc="Training Iterations", unit="batch")
  for epoch in range(config["max_epoch"]):
    model.train()

    # Log LR
    wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=iteration)

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
      pbar.update(1)
      iteration+=1


    # Evaluate on held out data
    val_loss, val_acc = evaluate(model, val_loader)
    wandb.log({"Loss/val": val_loss, "Acc/val": val_acc}, step=iteration)


    ################################
    #Model Checkpointing Q9 -save model if we hit new best validation accurcy
    ################################
    if val_acc > best_val:
      best_val = val_acc
      torch.save(model.state_dict(), "chkpts/" + run_name + "_epoch" + str(epoch))

    # Plot the predictions for a single test batch
    f = generatePredictionPlot(model, val_loader)
    wandb.log({"Viz/val":f}, step=iteration)
    plt.close()

    # Adjust LR
    scheduler.step()

  wandb.finish()
  pbar.close()


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
  run_name = ""+random_string+"_CIFAR10"
  return run_name

def generatePredictionPlot(model, test_loader):
  model.eval()
  x,y = next(iter(test_loader))
  out = F.softmax(model(x.to(device)).detach(), dim=1)

  num = min(20, x.shape[0])
  f, axs = plt.subplots(2, num, figsize=(4*num,8))
  for i in range(0,num):
    axs[0,i].imshow(x[i,:,:].squeeze().permute(1,2,0).cpu()*0.5+0.5)
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)
    axs[1,i].bar(list(range(10)),out[i,:].squeeze().cpu(), label=list(range(10)))
    axs[1,i].set_xticks([x-0.5 for x in range(10)], labels=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"],rotation=50)
    axs[1,i].set_ylim(0,1)

  return f


main()
