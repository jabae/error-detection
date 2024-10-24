# Import necessary packages
import numpy as np
from time import time

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F

from errordetector.augment.augmentor import Augmentor

from errordetector.utils.utils import *


def worker_init_fn(worker_id):
  
  # Each worker already has its own random state (Torch).
  seed = torch.IntTensor(1).random_()[0]
  np.random.seed(seed)


class Dataset(torch.utils.data.Dataset):
  
  def __init__(self, multidataset, vol_range, patch_size, out_size, aug_params):
    
    super(Dataset, self).__init__()

    full_labels_truth = MultiVolume(multidataset.human_labels[vol_range], patch_size)
    full_labels_lies = MultiVolume(multidataset.machine_labels[vol_range], patch_size)
    full_image = MultiVolume(multidataset.image[vol_range], patch_size)
    samples = MultiVolume(multidataset.samples[vol_range], (1,3), indexing='CORNER')

    self.full_labels_truth = full_labels_truth
    self.full_labels_lies = full_labels_lies
    self.full_image = full_image
    self.samples = samples

    augmentor = Augmentor(aug_params)
    self.augmentor = augmentor
    
    self.vol_size = multidataset.human_labels[0].shape
    self.patch_size = patch_size
    self.out_size = out_size 

    n = []
    for i in range(len(multidataset.samples[vol_range])):
      n.append(multidataset.samples[vol_range][i].shape[0])
        
    self.n = n
    self.n_cum = np.array([sum(n[:i+1]) for i in range(len(n))])
    self.size = sum(n)

  def __len__(self):

    return self.size

  def __getitem__(self, idx):

    n = self.n
    n_cum = self.n_cum
    vol_id = np.where((n_cum-idx)>0)[0][0]
    sid = n[vol_id]-(n_cum[vol_id]-idx)

    vol_size = self.vol_size
    patch_size = self.patch_size
    out_size = self.out_size

    focus = self.samples[vol_id, (sid,0)]
    focus = torch.cat((torch.zeros((1,), dtype=focus.dtype), torch.reshape(focus, (3,))), 0).numpy()

    labels_truth = self.full_labels_truth[vol_id, focus]
    labels_lies = self.full_labels_lies[vol_id, focus]
    image = self.full_image[vol_id, focus]
    
    obj_label = object_mask(labels_truth)
    obj_mask = object_mask(labels_lies)
    
    error = has_error(obj_mask, labels_truth, out_size)

    occluded = random_occlusion(obj_mask)
    
    sample = {"image": image, "reconstruct": obj_label, "obj_mask": obj_mask, "occluded": occluded, "error": error}

    # Augmentation
    for k in sample.keys():
      sample[k] = sample[k].numpy()            
    sample = self.augmentor(sample)
    for k in sample.keys():
      sample[k] = torch.from_numpy(sample[k].copy())

    return sample


class Data(object):
  
  def __init__(self, data, vol_ids, aug, opt, is_train=True):
    
    self.build(data, vol_ids, aug, opt, is_train)

  def __call__(self):
    
    sample = next(self.dataiter)
    for k in sample:
      is_input = k in self.inputs
      sample[k].requires_grad_(is_input)
      sample[k] = sample[k].cuda(non_blocking=(not is_input))

    return sample

  def requires_grad(self, key):
        
    return self.is_train and (key in self.inputs)

  def build(self, data, vol_ids, aug, opt, is_train):

    padded_patch_size = [1,] + opt.patch_size
    padded_out_size = [1,] + opt.out_size

    vol_range = slice(min(vol_ids), max(vol_ids)+1)

    aug_params = {'flip': False, 'rotate90': False, 'contrast': False}
    for k in aug:
      aug_params[k] = True

    dataset = Dataset(data, vol_range, padded_patch_size, padded_out_size, aug_params)       

    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)

    # Attributes
    self.dataiter = iter(dataloader)
    self.inputs = ['image', 'obj_mask']
    self.is_train = is_train