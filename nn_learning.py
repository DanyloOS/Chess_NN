#!pip install peewee pytorch-lightning
#!pip install tensorflow
#!pip install tensorboard

from peewee import *
import base64
import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, random_split
import pytorch_lightning as pl
from random import randrange
import time
from collections import OrderedDict
import chess

DB_PATH='test.db'

db = SqliteDatabase(DB_PATH)


class Evaluations(Model):
  id = IntegerField()
  fen = TextField()
  binary = BlobField()
  eval = FloatField()

  class Meta:
    database = db

  def binary_base64(self):
    return base64.b64encode(self.binary)
  

class EvaluationDataset(IterableDataset):
  def __init__(self, count):
    self.count = count
  def __iter__(self):
    return self
  def __next__(self):
    idx = randrange(self.count)
    return self[idx]
  def __len__(self):
    return self.count
  def __getitem__(self, idx):
    eval = Evaluations.get(Evaluations.id == idx+1)
    bin = np.frombuffer(eval.binary, dtype=np.uint8)
    bin = np.unpackbits(bin, axis=0).astype(np.single)
    eval.eval = max(eval.eval, -100)
    eval.eval = min(eval.eval, 100)
    ev = np.array([eval.eval]).astype(np.single)
    return {'binary':bin, 'eval':ev}

dataset = EvaluationDataset(count=LABEL_COUNT)


print(dataset.__next__())
print(dataset.__next__())
db.connect()
LABEL_COUNT = 37164639
print(LABEL_COUNT)
eval = Evaluations.get(Evaluations.id == 1)
print(eval.binary_base64())
print(Evaluations.get(Evaluations.id == 2).binary_base64())
