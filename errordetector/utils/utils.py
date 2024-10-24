# Import necessary packages
import numpy as np
import itertools
import operator
import h5py

import torch
from torch.nn.parallel import data_parallel
from torch.cuda import *


def h5read(filename):
    
  print("Loading " + filename + "...")

  f = h5py.File(filename, "r")
  img = f["main"][()]
  f.close()

  print("Loading complete!")

  return img


def h5write(filename, img):

  f = h5py.File(filename, "w")
  dset = f.create_dataset("main", data=img)
  f.close()
  print("Complete!")


class Dataset():

  def __init__(self, directory, d):
        
    self.directory = directory

    for (label, name) in d.items():
      setattr(self, label, prep(label, h5read(os.path.join(directory, name))))


class MultiDataset():

  def __init__(self, directories, d):
        
    self.n = len(directories)
    self.directories = directories
        
    for (label, name) in d.items():
      setattr(self, label, [prep(label, h5read(os.path.join(directory, name))) for directory in directories])


def prep(dtype, data):
    
  if dtype in ["image", "pred_errormap"]:
    img = autopad(data.astype(np.float32))
        
    if img.max() > 10:
      return img/255

    else:
      return img

  elif dtype in ["human_labels", "machine_labels", "labels"]:
    return autopad(data.astype(np.int32))

  elif dtype in ["labels64"]:
    return autopad(data.astype(np.int64))

  elif dtype in ["valid"]:
    return data.astype(np.int32)

  elif dtype in ["samples"]:
    return data.astype(np.int32)

  elif dtype in ["visited"]:
    return autopad(data.astype(np.int16))


def autopad(img):
    
  if len(img.shape) == 3:
    return np.reshape(img, (1,)+img.shape)

  elif len(img.shape) == 4:
    return np.reshape(img, img.shape)

  else:
    raise Exception("Autopad not applicable.")


def inrange(idx, size):

  if idx < 0:
    idx = 0

  elif idx >= size:
    idx = size-1

  return idx 


def combinations(elements):

  return np.array(list(itertools.product(elements[0], elements[1], elements[2])))


def one_hot(indices):

  if indices == 0:
    output = torch.tensor([1.,0.,0.])

  elif indices == 1:
    output = torch.tensor([0.,1.,0.])

  elif indices == 2:
    output = torch.tensor([0.,0.,1.])

  return output


def random_occlusion(target):

  patch_size = list(target.shape[1:])
  reshaped_target = torch.reshape(target, patch_size)

  xmask = torch.cat([torch.ones(patch_size[0]-int(patch_size[0]/2), patch_size[1], patch_size[2]), torch.zeros(int(patch_size[0]/2), patch_size[1], patch_size[2])], dim=0)
  ymask = torch.cat([torch.ones(patch_size[0], patch_size[1]-int(patch_size[1]/2), patch_size[2]), torch.zeros(patch_size[0], int(patch_size[1]/2), patch_size[2])], dim=1)
  zmask = torch.cat([torch.ones(patch_size[0], patch_size[1], patch_size[2]-int(patch_size[2]/2)), torch.zeros(patch_size[0], patch_size[1], int(patch_size[2]/2))], dim=2)
  full = torch.ones(patch_size)

  xmasks = torch.stack([xmask, 1-xmask, full])
  ymasks = torch.stack([ymask, 1-ymask, full])
  zmasks = torch.stack([zmask, 1-zmask, full])

  xweight = torch.zeros((3,))
  yweight = torch.zeros((3,))
  zweight = torch.zeros((3,))

  for i in range(3):
    xweight[i] = torch.exp(0.3*torch.log(0.001+torch.sum(xmasks[i,...]*torch.stack([reshaped_target]))))
    yweight[i] = torch.exp(0.3*torch.log(0.001+torch.sum(ymasks[i,...]*torch.stack([reshaped_target]))))
    zweight[i] = torch.exp(0.3*torch.log(0.001+torch.sum(zmasks[i,...]*torch.stack([reshaped_target]))))

  xchoice = torch.reshape(one_hot(torch.multinomial(xweight, 1)),(3,1,1,1))
  ychoice = torch.reshape(one_hot(torch.multinomial(yweight, 1)),(3,1,1,1))
  zchoice = torch.reshape(one_hot(torch.multinomial(zweight, 1)),(3,1,1,1))
  
  mask = torch.sum(xmasks*xchoice, dim=0)*torch.sum(ymasks*ychoice, dim=0)*torch.sum(zmasks*zchoice, dim=0)
  mask = torch.reshape(mask, [1]+patch_size)

  return mask*target     


def random_coord_valid(volume_size, patch_size, n=1):

  x = np.random.randint(low=patch_size[0]//2, high=volume_size[0]-patch_size[0]//2, size=n)
  y = np.random.randint(low=patch_size[1]//2, high=volume_size[1]-patch_size[1]//2, size=n)
  z = np.random.randint(low=patch_size[2]//2, high=volume_size[2]-patch_size[2]//2, size=n)

  x = np.reshape(x, [x.size,-1])
  y = np.reshape(y, [y.size,-1])
  z = np.reshape(z, [z.size,-1])

  random_coord = np.concatenate([x,y,z], axis=1)

  return random_coord


def object_mask(img):

  shape = img.shape
  if len(shape) == 3:
    obj_id = img[shape[0]//2, shape[1]//2, shape[2]//2]
  elif len(shape) == 4:
    obj_id = img[shape[0]//2, shape[1]//2, shape[2]//2, shape[3]//2]
    
  if isinstance(img, (torch.Tensor)):    
    mask = (img==obj_id).float()
  else:
    mask = (img==obj_id).astype(np.float32)

  return mask


class Volume():

  def __init__(self, A, patch_size, indexing='CENTRAL'):

    self.A = A
    self.patch_size = patch_size
    self.indexing = indexing

  def __getitem__(self, focus):

    A = self.A
    patch_size = self.patch_size

    if type(focus) == tuple:
      focus = list(focus)
        
    for i, s in enumerate(A.shape):
      if focus[i] == 'RAND':
        focus[i] = np.random.randint(0,s,[])

      if self.indexing == 'CENTRAL':
        corner = focus - np.array([x/2 for x in patch_size], dtype=np.int32)
        corner = np.reshape(corner,(-1,))

      elif self.indexing == 'CORNER':
        corner = focus

      else:
        raise Exception("Bad indexing scheme.")


    return A[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])]

  def __setitem__(self, focus, val):
        
    patch_size = self.patch_size

    if self.indexing == 'CENTRAL':
      corner = focus - np.array([x/2 for x in patch_size], dtype=np.int32)
      corner = np.reshape(corner,(-1,))

    elif self.indexing == 'CORNER':
      corner = focus

    else:
      raise Exception("Bad indexing scheme.")

    self.A[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])] = val


class MultiVolume():

  def __init__(self, As, patch_size, indexing = 'CENTRAL'):
         
    self.As = list(map(lambda A: Volume(A, patch_size, indexing=indexing), As))
    self.patch_size = patch_size

  def __getitem__(self, index):

    vol_index, focus = index

    if vol_index <= len(list(self.As))-1:
      vol_out = torch.tensor(self.As[vol_index][focus])

    else:
      vol_out = torch.tensor(self.As[0][focus])
      print(str(vol_index) + " volume not found.")

    return torch.reshape(vol_out, self.patch_size)

  def __setitem__(self, index, val):

    vol_index, focus = index

    self.As[vol_index][focus] = val
        

def error_free(obj, human_labels, out_size):

  in_size = np.array(obj.shape)
  out_size = np.array(out_size)
  patch_size = np.array([1,3,46,46])
  padded_in_size = in_size + 2*(patch_size//2)

  error = np.ones(tuple(out_size), dtype=np.float32)

  obj_reshape = np.reshape(obj.numpy(), (-1,))
  human_labels_reshape = np.reshape(human_labels, (-1,))

  closest_obj = np.array(human_labels_reshape == human_labels_reshape[np.argmax(obj_reshape)], dtype=np.float32)

  error_vol = np.min(obj_reshape==closest_obj)

  # Initial check whether patch has error 
  if error_vol == 1:
    return torch.tensor(error, dtype=torch.float32)

  if max(out_size) == 1:        
    error[0,0,0,0] = error_vol

  else: 
    overlap = np.ceil((patch_size[1:]*out_size[1:]-padded_in_size[1:])/(out_size[1:]-1)).astype(int)
        
    for i in range(out_size[1]):
      for j in range(out_size[2]):
        for k in range(out_size[3]):

          x_min = max((patch_size[1]-overlap[0])*i-patch_size[1]//2, 0) 
          x_max = (patch_size[1]-overlap[0])*i+patch_size[1]-patch_size[1]//2
          y_min = max((patch_size[2]-overlap[1])*j-patch_size[2]//2, 0)
          y_max = (patch_size[2]-overlap[1])*j+patch_size[2]-patch_size[2]//2
          z_min = max((patch_size[3]-overlap[2])*k-patch_size[3]//2, 0)
          z_max = (patch_size[3]-overlap[2])*k+patch_size[3]-patch_size[3]//2

          obj_chunk = obj[0, x_min:x_max, y_min:y_max, z_min:z_max]
          if torch.sum(obj_chunk) == 0:
            continue

          human_labels_chunk = human_labels[0, x_min:x_max, y_min:y_max, z_min:z_max].numpy()

          obj_chunk = np.reshape(obj_chunk.numpy(), (-1,))
          human_labels_chunk = np.reshape(human_labels_chunk, (-1,))
          
          closest_obj = np.array(human_labels_chunk == human_labels_chunk[np.argmax(obj_chunk)], dtype=np.float32)

          error[0,i,j,k] = np.min(obj_chunk==closest_obj)
 
    
  return torch.tensor(error, dtype=torch.float32)


def has_error(obj, human_labels, out_size):

  return 1 - error_free(obj, human_labels, out_size)


def has_merge_error(obj, human_labels, out_size):

    in_size = np.array(obj.shape)
    out_size = np.array(out_size)
    patch_size = np.array([1,3,46,46])
    padded_in_size = in_size + 2*(patch_size//2)

    error = np.zeros((1,)+tuple(out_size[1:]), dtype=np.float32)

    obj_reshape = np.reshape(obj.numpy(), (-1,))
    human_labels_reshape = np.reshape(human_labels, (-1,))

    closest_obj = np.array(human_labels_reshape == human_labels_reshape[np.argmax(obj_reshape)], dtype=np.float32)

    error_vol = np.min(obj_reshape==closest_obj)

    # Initial check whether patch has error 
    if error_vol == 1:
        return torch.tensor(error, dtype=torch.float32)

    if max(out_size) == 1:
        if np.sum(closest_obj) > np.sum(obj_reshape):        
            error[0,0,0,0] = 0
        else:
            error[0,0,0,0] = 1

    else: 
        overlap = np.ceil((patch_size[1:]*out_size[1:]-padded_in_size[1:])/(out_size[1:]-1)).astype(int)
        
        for i in range(out_size[1]):
            for j in range(out_size[2]):
                for k in range(out_size[3]):

                    x_min = max((patch_size[1]-overlap[0])*i-patch_size[1]//2, 0) 
                    x_max = (patch_size[1]-overlap[0])*i+patch_size[1]-patch_size[1]//2
                    y_min = max((patch_size[2]-overlap[1])*j-patch_size[2]//2, 0)
                    y_max = (patch_size[2]-overlap[1])*j+patch_size[2]-patch_size[2]//2
                    z_min = max((patch_size[3]-overlap[2])*k-patch_size[3]//2, 0)
                    z_max = (patch_size[3]-overlap[2])*k+patch_size[3]-patch_size[3]//2

                    obj_chunk = obj[0, x_min:x_max, y_min:y_max, z_min:z_max]
                    if torch.sum(obj_chunk) == 0:
                        continue

                    human_labels_chunk = human_labels[0, x_min:x_max, y_min:y_max, z_min:z_max].numpy()

                    obj_chunk = np.reshape(obj_chunk.numpy(), (-1,))
                    human_labels_chunk = np.reshape(human_labels_chunk, (-1,))
                    
                    closest_obj = np.array(human_labels_chunk == human_labels_chunk[np.argmax(obj_chunk)], dtype=np.float32)

                    no_error = np.min(obj_chunk==closest_obj)

                    if ~no_error and np.sum(closest_obj) > np.sum(obj_chunk): 
                        error[0,i,j,k] = 0
                    elif ~no_error and np.sum(closest_obj) < np.sum(obj_chunk):
                        error[0,i,j,k] = 1

    return torch.tensor(error, dtype=torch.float32)


def precision_recall(label, pred_prob, thresholds):

  precision = np.zeros(thresholds.shape)
  recall = np.zeros(thresholds.shape)
  F1 = np.zeros(thresholds.shape)

  for i in range(len(thresholds)):

    thr = thresholds[i]

    pos = np.sum(pred_prob>=thr)

    if pos == 0:
      precision[i] = np.inf
      recall[i] = 0
      F1[i] = np.inf

    else: 
      true_pos = np.sum(label*(pred_prob>=thr))
      precision[i] = true_pos/pos
      recall[i] = true_pos/np.sum(label==1)
      F1[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])


  return (precision, recall, F1, thresholds)


def stat_summary(label, pred_prob, thresholds):

  accuracy = np.zeros(thresholds.shape)
  precision = np.zeros(thresholds.shape)
  recall = np.zeros(thresholds.shape)
  F1 = np.zeros(thresholds.shape)

  for i in range(len(thresholds)):

    thr = thresholds[i]
    pred = pred_prob>=thr
    pos = np.sum(pred)
    
    accuracy[i] = np.sum(label==pred)/label.shape[0]

    if pos == 0:
      precision[i] = np.inf
      recall[i] = 0
      F1[i] = np.inf

    else:
      true_pos = np.sum(label*pred)
      precision[i] = true_pos/pos
      recall[i] = true_pos/np.sum(label==1)
      F1[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])


  return (accuracy, precision, recall, F1, thresholds)
