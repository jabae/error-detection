# Import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F


avgpool = nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)
maxpool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)

class Model(nn.Module):
  """
  Model wrapper for training.
  """
  def __init__(self, model, mip=1):

    super(Model, self).__init__()
    self.model = model
    self.mip = mip

  def forward(self, sample):

    mip = self.mip

    # 2-channel input (object mask/raw EM)
    inputs_list = []
    for k in ['image','obj_mask']:
      inputs_list.append(sample[k].cuda())
    
    inputs = torch.cat(inputs_list, dim=1)

    # Forward pass
    preds = {}

    # Error detection
    for i in range(mip):
      inputs = avgpool(inputs)
        
    out = self.model.discrim(inputs)
    preds["error"] = torch.sigmoid(out)
        
    return preds

  def save(self, fpath):
  
    torch.save(self.model.state_dict(), fpath)

  def load(self, fpath):

    state_dict = torch.load(fpath)
        
    self.model.load_state_dict(state_dict,strict=False)
