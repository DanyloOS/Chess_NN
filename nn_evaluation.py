#!/home/danylos/anaconda3/bin/python

# import sys
# import threading
# import cmd
import chess
# from chess import polyglot
# import tables

# import os
import torch
import numpy as np
from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, IterableDataset, random_split
import pytorch_lightning as pl
# from random import randrange

# import time
from collections import OrderedDict

network_file_name="model_state_dicts_trained_well"
database_path="test.db"
configs = [
           #{"layer_count": 4, "batch_size": 512},
            {"layer_count": 6, "batch_size": 1024},
           ]

############################

from peewee import *
import base64

##################

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




model = EvaluationModel(layer_count=6,batch_size=1024,learning_rate=1e-3)
model.load_state_dict(torch.load(network_file_name))


def bitboard(board, debug=False):
    w_pawn = (np.asarray(board.pieces(chess.PAWN, chess.WHITE).tolist())).astype(int)
    w_rook = (np.asarray(board.pieces(chess.ROOK, chess.WHITE).tolist())).astype(int)
    w_knight = (np.asarray(board.pieces(chess.KNIGHT, chess.WHITE).tolist())).astype(int)
    w_bishop = (np.asarray(board.pieces(chess.BISHOP, chess.WHITE).tolist())).astype(int)
    w_queen = (np.asarray(board.pieces(chess.QUEEN, chess.WHITE).tolist())).astype(int)
    w_king = (np.asarray(board.pieces(chess.KING, chess.WHITE).tolist())).astype(int)

    b_pawn = (np.asarray(board.pieces(chess.PAWN, chess.BLACK).tolist())).astype(int)
    b_rook = (np.asarray(board.pieces(chess.ROOK, chess.BLACK).tolist())).astype(int)
    b_knight = (np.asarray(board.pieces(chess.KNIGHT, chess.BLACK).tolist())).astype(int)
    b_bishop = (np.asarray(board.pieces(chess.BISHOP, chess.BLACK).tolist())).astype(int)
    b_queen = (np.asarray(board.pieces(chess.QUEEN, chess.BLACK).tolist())).astype(int)
    b_king = (np.asarray(board.pieces(chess.KING, chess.BLACK).tolist())).astype(int)
    
    testFen = board.fen(en_passant="fen")
    if debug:
        print(testFen)
    bit12WhoToMove = 1 if testFen.split()[1] == "b" else 0
    numericLetter = -1
    rankNumber = 0
    if '-' not in testFen.split()[3]:
        numericLetter = ord(testFen.split()[3][0]) - ord('a')
        rankNumber = int(testFen.split()[3][1]) - 1
    
    isEnPassant = 0 if testFen.split()[3][0] == '-' else 1
    castlingStr = testFen.split()[2]
    castling = ((1<<3) if "q" in castlingStr else 0) + ((1<<2) if "k" in castlingStr else 0)\
             + ((1<<1) if "Q" in castlingStr else 0) + (1 if "K" in castlingStr else 0)
    
    firstInt = int(testFen.split()[-2])
    secondInt = 0
    thirdInt = int(testFen.split()[-1])
    forthInt = (rankNumber << 3) + numericLetter
    fifthInt = (isEnPassant << 5) + (bit12WhoToMove << 4) + castling
    
    # add_info = np.array([firstInt, secondInt, thirdInt, forthInt, fifthInt], dtype=np.uint8)
    add_info = np.array([firstInt, secondInt, thirdInt, forthInt, fifthInt]).astype(np.uint8)
    add_info = np.unpackbits(add_info, axis=0).astype(np.single) 

    customBitboard = np.concatenate((w_king, w_queen, w_rook, w_bishop, w_knight, w_pawn,
                                     b_king, b_queen, b_rook, b_bishop, b_knight, b_pawn,
                                     add_info))
    
    if debug:
      print(customBitboard)
    return customBitboard

#################################

def testBitboard(idx):
    # eval1 = Evaluations.select().where(Evaluations.id == idx + 1).get()
    # evall_fen = evall.fen
    evall_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(evall_fen)
    board.set_fen(evall_fen)
    # myBitboard = bitboard(board, debug=True)
    myBitboard = bitboard(board, debug=False)
    # print()
    # print("If no lines bellow then everything is ok!")

    # if False in (dataset[idx]["binary"] == myBitboard):
        # print(idx, dataset[idx]["binary"] == myBitboard)
        # print(myBitboard[-16:])
        # print(dataset[idx]["binary"][-16:])
        # print(evall_fen)
    


def positionScore(fen):
    target_bitboard = bitboard(chess.Board(fen))
    x = torch.tensor(target_bitboard).float()
    y_hat = model(x)
    print(f'FEN {fen}')
    print(f'Prediction {y_hat.data[0]:.2f}')
    # display(SVG(url=svg_url(fen)))
    return y_hat.data[0]
    

    
def boardScore(board):
    target_bitboard = bitboard(board)
    x = torch.tensor(target_bitboard).float()
    y_hat = model(x)
    # print(f'FEN {board.fen(en_passant="fen")}')
    # print(f'Prediction {y_hat.data[0]:.2f}')
    # display(SVG(url=svg_url(fen)))

    
    return y_hat.data[0]
    


testBitboard(1)
print("hellno")
positionScore("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")