# Import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F

from errordetector.utils.utils import random_occlusion


class Model(nn.Module):
  """
  Model wrapper for training.
  """
  def __init__(self, model, opt):
  
    super(Model, self).__init__()
    self.model = model
    self.in_spec = opt.in_spec
    self.out_spec = opt.out_spec
    self.pretrain = opt.pretrain is not None
    self.mip = opt.mip

    mip_factor = 2**opt.mip
    kernel_size = (1,mip_factor,mip_factor)
    self.avgpool = nn.AvgPool3d(kernel_size=kernel_size, stride=kernel_size, padding=0)
    self.maxpool = nn.MaxPool3d(kernel_size=kernel_size, stride=kernel_size, padding=0)
    self.upsample = nn.Upsample(scale_factor=kernel_size, mode='trilinear', align_corners=True)

  def forward(self, sample):
  
    in_spec = self.in_spec
    mip = self.mip
    avgpool = self.avgpool
    maxpool = self.maxpool
    upsample = self.upsample

    # Downsample
    for k in in_spec:
      if k in ["obj_mask", "occluded"]:
        sample[k] = maxpool(sample[k])
      elif k in ["image"]:
        sample[k] = avgpool(sample[k])

    preds = {}
      
    # Reconstruct
    inputs_list = [sample["image"],sample["occluded"]]
    inputs = torch.cat(inputs_list, dim=1)
   
    reconstruct = self.model(inputs)
    reconstruct = upsample(reconstruct)

    preds["reconstruct"] = reconstruct

    # Error detection
    inputs_list = [sample["image"],sample["obj_mask"]]
    inputs = torch.cat(inputs_list, dim=1)

    # Error detection
    out = self.model.discrim(inputs)
      
    preds["error"] = out

    # Loss evaluation
    losses = self.eval_loss(preds, sample)
      
    return losses, preds

  def eval_loss(self, preds, sample):
      
    losses = dict()

    for k in self.out_spec:
  
      loss = F.binary_cross_entropy_with_logits(input=preds[k], target=sample[k])
      losses[k] = loss.unsqueeze(0)
      
    return losses

  def save(self, fpath):

    torch.save(self.model.state_dict(), fpath)

  def load(self, fpath):

    state_dict = torch.load(fpath)
    if self.pretrain:
      model_dict = self.model.state_dict()
      state_dict = {k:v for k, v in state_dict.items() if k in model_dict}
      model_dict.update(state_dict)
      self.model.load_state_dict(model_dict)
    else:
      self.model.load_state_dict(state_dict)
