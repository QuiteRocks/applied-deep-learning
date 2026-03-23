from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch


def getCIFAR10Dataloaders(config):

  # Set up dataset and data loaders
  test_transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
      ])

  
  train_transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(),
      transforms.RandomResizedCrop((32,32)),
      transforms.ColorJitter(brightness=0.25, hue=0.25),
      transforms.Normalize((0.5,), (0.5,))
      ])
 

  train_set = datasets.CIFAR10(root='./datasets/data', train=True,download=True, transform=train_transform)
  val_set = datasets.CIFAR10(root='./datasets/data', train=True,download=True, transform=test_transform)

  indices = torch.randperm(len(train_set))
  val_size = len(train_set)//8
  train_set = Subset(train_set, indices[:-val_size])
  val_set = Subset(val_set, indices[-val_size:])

  test_set = datasets.CIFAR10(root='./datasets/data', train=False,download=True, transform=test_transform)


  train_loader = DataLoader(train_set, shuffle=True, batch_size=config["bs"], num_workers=4, pin_memory=True)
  val_loader = DataLoader(val_set, shuffle=False, batch_size=5*config["bs"], num_workers=4, pin_memory=True)
  test_loader = DataLoader(test_set,  shuffle=False, batch_size=config["bs"])

  return train_loader, val_loader, test_loader