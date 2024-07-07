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


class EvaluationModel(pl.LightningModule):
  def __init__(self,learning_rate=1e-3,batch_size=1024,layer_count=10):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    layers = []
    for i in range(layer_count-1):
      layers.append((f"linear-{i}", nn.Linear(808, 808)))
      layers.append((f"relu-{i}", nn.ReLU()))
    layers.append((f"linear-{layer_count-1}", nn.Linear(808, 1)))
    self.seq = nn.Sequential(OrderedDict(layers))

  def forward(self, x):
    return self.seq(x)

  def training_step(self, batch, batch_idx):
    x, y = batch['binary'], batch['eval']
    y_hat = self(x)
    loss = F.l1_loss(y_hat, y)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

  def train_dataloader(self):
    dataset = EvaluationDataset(count=LABEL_COUNT)
    return DataLoader(dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

if __name__=="__main__":
    LABEL_COUNT = 37164639
    dataset = EvaluationDataset(count=LABEL_COUNT)

    configs = [
            #  {"layer_count": 4, "batch_size": 64},
            #  {"layer_count": 4, "batch_size": 512},
            {"layer_count": 5, "batch_size": 512},
            #  {"layer_count": 6, "batch_size": 1024},
            ]
    for config in configs:
        torch.set_float32_matmul_precision('medium')
        version_name = f'{int(time.time())}-batch_size-{config["batch_size"]}-layer_count-{config["layer_count"]}'
        logger = pl.loggers.TensorBoardLogger("lightning_logs", name="chessml", version=version_name)
        trainer = pl.Trainer(precision="16-mixed",max_epochs=1,logger=logger)

    model = EvaluationModel(layer_count=config["layer_count"],batch_size=config["batch_size"],learning_rate=1e-3)

    trainer.fit(model)
    torch.save(model.state_dict(), "model_state_dicts_layer5_batch512_1e3_epochs1111")
    # model = EvaluationModel(layer_count=config["layer_count"],batch_size=config["batch_size"],learning_rate=1e-3)
    # model.load_state_dict(torch.load('model_state_dicts_layer5_batch512_1e3_epochs1'))

    print(dataset.__next__())
    print(dataset.__next__())
    db.connect()
    LABEL_COUNT = 37164639
    print(LABEL_COUNT)
    eval = Evaluations.get(Evaluations.id == 1)
    print(eval.binary_base64())
    print(Evaluations.get(Evaluations.id == 2).binary_base64())
