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
from datetime import datetime


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
    # eval.eval = max(eval.eval, -100)
    # eval.eval = min(eval.eval, 100)
    eval.eval = max(eval.eval, -10)
    eval.eval = min(eval.eval, 10)
    ev = np.array([eval.eval]).astype(np.single)
    return {'binary':bin, 'eval':ev}



LABEL_COUNT = 37164639
dataset = EvaluationDataset(count=LABEL_COUNT)

eval = Evaluations.get(100)
print(eval.fen, eval.eval)
# print(dataset.__next__())
# print(dataset.__next__())
# # db.connect()
# LABEL_COUNT = 37164639
# print(LABEL_COUNT)
# eval = Evaluations.get(Evaluations.id == 1)
# print(eval.binary_base64())
# print(Evaluations.get(Evaluations.id == 2).binary_base64())
# Field names
field_names = ('FEN', 'original_db prediction')

import csv
import sys

# Open a file for writing
with open(f'original_db prediction.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    
    # Write header
    writer.writerow(field_names) 
    
    # Write rows
    while True:
        fen_index = input()
        if fen_index == "":
          continue
        
        if fen_index == "end":
          break

        cur_possition = Evaluations.get(int(fen_index) + 1)
        print(f'Prediction {cur_possition.eval}')
        writer.writerow((cur_possition.fen, cur_possition.eval))  
        # print(f'FEN {fen_position}')
