import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


class Vocab:
  def __init__(self, data):
    unique_words = Counter()
    labels = Counter()
    for x,y in data:
      unique_words.update(x)
      labels.update(y)
    
    self.word2idx = {}
    self.idx2word = {}
    for idx, word in enumerate(unique_words):
      self.word2idx[word] = idx+2
      self.idx2word[idx+2] = word

    self.word2idx["<UNK>"] = 0
    self.idx2word[0] = "<UNK>"

    self.word2idx["<PAD>"] = 1
    self.idx2word[1] = "<PAD>"

    self.label2idx = {}
    self.idx2label = {}
    for idx, word in enumerate(labels):
      self.label2idx[word] = idx
      self.idx2label[idx] = word
  
  def lenWords(self):
    return len(self.word2idx.items())

  def lenLabels(self):
    return len(self.label2idx.items())

  def numeralizeLabels(self, words):
    return [self.label2idx[w] for w in words]

  def denumeralizeLabels(self, idxs):
    return [self.idx2label[i] for i in idxs]

  def numeralizeSentence(self, words):
    out = []
    for w in words:
      if w in self.word2idx.keys():
        out.append(self.word2idx[w])
      else:
        out.append(self.word2idx["<UNK>"])
    return out

  def denumeralizeSentence(self, idxs):
    out = []
    for i in idxs:
      if i in self.idx2word.keys():
        out.append(self.idx2word[i])
      else:
        out.append("<UNK>")
    return out
    


class UDPOSDataset(Dataset):
  def __init__(self, split="train", vocab=None):
    super().__init__()
    assert( split == "train" or vocab != None) # must provide a vocab for non-train splits

    self.data = self.loadData(split)
    if vocab != None:
      self.vocab = vocab
    else:
      self.vocab = Vocab(self.data)

  def loadData(self, split):
    data = []
    x = []
    y = []
    filename = "./data/UDPOS/en-ud-tag.v2."+split+".txt"
    with open(filename, "r", encoding="utf-8") as f:  
      for line in f:
        if line == "\n":
          data.append( (x,y) )
          x = []
          y = []
        else:
          s = line.split('\t')
          if s[1] != "X":
            x.append(s[0].lower())
            y.append(s[1])
    return data

  

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    x ,y = self.data[idx]

    x = torch.LongTensor(self.vocab.numeralizeSentence(x))
    y = torch.LongTensor(self.vocab.numeralizeLabels(y))

    return x, y
  
  # Function to enable batch loader to stack binary strings of different lengths and pad them
  @staticmethod
  def pad_collate(batch):# Q4
    ###########################################
    #
    xs, ys = zip(*batch)  #unzip the batch into separet lists of x sequences and y lables 
    x_len = torch.LongTensor([len(x) for x in xs])
    xx = pad_sequence(list(xs), batch_first=True, padding_value=1)
    yy = pad_sequence(list(ys), batch_first=True, padding_value=-1)
    return xx, yy, x_len
    #
    ###########################################


def getUDPOSDataloaders(batch_size=128):
   train_data = UDPOSDataset(split='train')
   val_data = UDPOSDataset(split='dev', vocab=train_data.vocab)
   test_data = UDPOSDataset(split='test', vocab=train_data.vocab)

   train_loader = DataLoader(dataset=train_data, batch_size=batch_size, 
              shuffle=True, num_workers=8, persistent_workers=True,
              drop_last=True, collate_fn=UDPOSDataset.pad_collate)
   
   val_loader = DataLoader(dataset=val_data, batch_size=batch_size, 
              shuffle=False, num_workers=8, persistent_workers=True,
              drop_last=False, collate_fn=UDPOSDataset.pad_collate)
   
   test_loader = DataLoader(dataset=test_data, batch_size=batch_size, 
              shuffle=False, num_workers=8, persistent_workers=True,
              drop_last=False, collate_fn=UDPOSDataset.pad_collate)

   return train_loader, val_loader, test_loader, train_data.vocab
