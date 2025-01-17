import numpy as np


def rotate90_augment(sample):
  """Performs rotation augmentation on img.
    Only do integer multiples of 90 deg. Only rotate about z-axis.

  Args:
    img: (np array: <ch,z,y,x>) image
    labels: list of (np array: <ch,z,y,x>), pixelwise labeling of img
  """
  # Only z rotation considering anisotropy.

  # z rotation  
  rotations = (0,1,2,3)
  i = np.random.choice(rotations)
  
  for k in sample.keys():  
    sample[k] = np.rot90(sample[k], k=i, axes=(2,3))
    
    
  return sample
