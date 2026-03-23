import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import matplotlib.pyplot as plt
import datetime
import random
import string
import pickle
import wandb
from tqdm import tqdm

# Import our own files
from data.PoSData import Vocab, getUDPOSDataloaders
from models.PoSGRU import PoSGRU

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
elif use_cuda_if_avail and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

config = {
    "bs":256,   # batch size
    "lr":0.0005, # learning rate
    "l2reg":0.0000001, # weight decay
    "max_epoch":30,
    "layers": 2,
    "embed_dim":128,
    "hidden_dim":256,
    "residual":True
}


def main():

  # Get dataloaders
  train_loader, val_loader, _, vocab = getUDPOSDataloaders(config["bs"])

  vocab_size = vocab.lenWords()
  label_size = vocab.lenLabels()

  # Build model
  model = PoSGRU(vocab_size=vocab_size, 
                 embed_dim=config["embed_dim"], 
                 hidden_dim=config["hidden_dim"], 
                 num_layers=config["layers"],
                 output_dim=label_size,
                 residual=config["residual"])
  print(model)

  torch.compile(model)


  # Start model training
  train(model, train_loader, val_loader)




def train(model, train_loader, val_loader):

  # Log our exact model architecture string
  config["arch"] = str(model)
  run_name = generateRunName()

  # Startup wandb logging
  wandb.login()
  wandb.init(project="UDPOS CS435 A6", name=run_name, config=config)

  # Move model to the GPU
  model.to(device)

  # Set up optimizer and our learning rate schedulers
  optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
  warmup_epochs = config["max_epoch"]//10
  linear = LinearLR(optimizer, start_factor=0.25, total_iters=warmup_epochs)
  cosine = CosineAnnealingLR(optimizer, T_max = config["max_epoch"]-warmup_epochs)
  scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs])

  # Loss
  ################### Q5 Loss#########################
  #
  crit = nn.CrossEntropyLoss(ignore_index=-1)
  #
  ###########################################

  best_val_acc = 0.0

  # Main training loop with progress bar
  iteration = 0
  pbar = tqdm(total=config["max_epoch"]*len(train_loader), desc="Training Batches", unit="batch")
  for epoch in range(config["max_epoch"]):
    model.train()

    # Log LR
    wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=iteration)

    for x, y, lens in train_loader:
      x = x.to(device)
      y = y.to(device)

      out = model(x)
      

      ##############Q5 Loss##############################
      #
      loss = crit(out.view(-1, out.size(-1)), y.view(-1))

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      #
      ###########################################
      
      ############### Q5 Accuracy#############################
      #
      preds = torch.argmax(out, dim=-1)
      mask = (y != -1) #ignore padding
      acc = (preds[mask] == y[mask]).float().mean()
      #
      ###########################################


      wandb.log({"Loss/train": loss.item(), "Acc/train": acc.item()}, step=iteration)
      pbar.update(1)
      iteration+=1

    val_loss, val_acc = evaluate(model, val_loader)
    wandb.log({"Loss/val": val_loss, "Acc/val": val_acc}, step=iteration)

    ################# Q6 Checkpointing###########################
    #
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'config': config,
      }, 'best_model.pt')
      with open('vocab.pkl', 'wb') as f:
        pickle.dump(train_loader.dataset.vocab, f)
      print("saved best model at epoch", epoch, "with val acc:", round(val_acc, 4))
    #
    ###########################################
      

    # Adjust LR
    scheduler.step()

  wandb.finish()
  pbar.close()


def evaluate(model, loader):# Q6 
  ###########################################
  #
  model.eval()

  crit = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
  running_loss = 0.0
  running_acc = 0.0
  nonpad = 0

  with torch.no_grad():
    for x, y, lens in loader:
      x = x.to(device)
      y = y.to(device)

      out = model(x)

      loss = crit(out.view(-1, out.size(-1)), y.view(-1))
      running_loss += loss.item()

      preds = torch.argmax(out, dim=-1)
      mask = (y != -1)
      running_acc += (preds[mask] == y[mask]).float().sum().item()
      nonpad += mask.sum().item()
  #
  ###########################################

  return running_loss/nonpad, running_acc/nonpad

def generateRunName():
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  now = datetime.datetime.now()
  run_name = ""+random_string+"_UDPOS"
  return run_name



main()