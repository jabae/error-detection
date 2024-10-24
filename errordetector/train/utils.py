# Import necessary packages
import os

import numpy as np

import torch
from torch.nn.parallel import data_parallel
from torch.cuda import *

from errordetector.train.data import Data
from errordetector.train.model import Model


def load_model(opt):

  # Create a model.
  net = opt.net
  model_dir = opt.model_dir
  chkpt_num = opt.chkpt_num
  pretrain = opt.pretrain

  net.cuda()
  model = Model(net, opt)

  if pretrain:
    model.load(pretrain)
  if chkpt_num > 0:
    model = load_chkpt(model, model_dir, chkpt_num)

  return model.train()


def load_chkpt(model, fpath, chkpt_num):
  
  print("LOAD CHECKPOINT: {} iters.".format(chkpt_num))
  fname = os.path.join(fpath, "model_{}.chkpt".format(chkpt_num))
  model.load(fname)

  return model


def save_chkpt(model, fpath, chkpt_num):
  
  print("SAVE CHECKPOINT: {} iters.".format(chkpt_num))
  fname = os.path.join(fpath, "model_{}.chkpt".format(chkpt_num))
  model.save(fname)


def load_data(dataset, vol, aug, opt):

  data_loader = Data(dataset, vol, aug, opt, is_train=True)

  return data_loader


def forward(model, sample, batch_size):
    
  # Forward pass
  if batch_size > 1:
    losses, preds = data_parallel(model, sample)
  
  else:
    losses, preds = model(sample)

  # Average over minibatch
  losses = {k: v.mean() for k, v in losses.items()}

  return losses, preds