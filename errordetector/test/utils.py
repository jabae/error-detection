# Import necessary packages
import os

import numpy as np

import torch
from torch.nn.parallel import data_parallel
from torch.cuda import *

from errordetector.test.data import Data
from errordetector.test.model import Model


def load_model(opt):

    net = opt.net
    model_dir = opt.model_dir
    chkpt_num = opt.chkpt_num
    mip = opt.mip

    net.cuda()
    model = Model(net, mip)

    if chkpt_num > 0:
        model = load_chkpt(model, model_dir, chkpt_num)

    return model.eval()


def load_data(dataset, vol, opt):

  patch_size = opt.patch_size
  out_size = opt.out_size

  data_loader = Data(opt, dataset, vol, patch_size, out_size, is_train=False)

  return data_loader


def load_chkpt(model, fpath, chkpt_num):
  
  print("LOAD CHECKPOINT: {} iters.".format(chkpt_num))
  fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
  model.load(fname)

  return model


def save_chkpt(model, fpath, chkpt_num):
    
  print("SAVE CHECKPOINT: {} iters.".format(chkpt_num))
  fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
  model.save(fname)


def forward(model, sample):
  # Forward pass    
  preds = model(sample)

  return preds
