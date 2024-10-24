import numpy as np

def flip_augment(sample):
  """Performs flip augmentation on img.

  Args:
    img: (np array: <ch,z,x,y>) image
  """

  # z flip
  if np.random.rand() < 0.5:
    for k in sample.keys():
      sample[k] = np.flip(sample[k], axis=1)

  # x flip
  if np.random.rand() < 0.5:
    for k in sample.keys():
      sample[k] = np.flip(sample[k], axis=2)

  # y flip
  if np.random.rand() < 0.5:
    for k in sample.keys():
      sample[k] = np.flip(sample[k], axis=3)


  return sample


