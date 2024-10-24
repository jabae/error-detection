"""Provides data augmentation"""
import numpy as np

from .flip import flip_augment
from .rotate90 import rotate90_augment
from .contrast import contrast_augment

class Augmentor:
  def __init__(self, params):
    self.params = params
    self._init_params()

  def _init_params(self):
    augs = ['flip', 'rotate90', 'contrast']
    for aug in augs:
      if aug not in self.params.keys():
        self.params[aug] = False

  def __call__(self, sample):
    return self.augment(sample)

  def augment(self, sample):
    """Augments example.

    Args:
      img: (np array: <z,y,x,ch>) image
      labels: list of (int np array: <z,y,x,ch>), pixelwise labeling of image
      params: dict containing augmentation parameters, see code for details

    Returns:
      augmented img: image after augmentation
      augmented labels: labels after augmentation

    Note:
      augmented img,labels may not be same size as input img,labels
        because of warping
    """
    params = self.params

    # Flip
    if params['flip']:
      sample = flip_augment(sample)

    # Rotate
    if params['rotate90']:
      sample = rotate90_augment(sample)

    # Contrast adjust
    if params['contrast']:
      sample = contrast_augment(sample)

    # if params['rotate']:
    #   mode = params['rotate_mode']
    #   img, labels = rotate_augment(img, labels, mode)

    # # Rescale
    # if params['rescale']:
    #   min_f = params['rescale_min']
    #   max_f = params['rescale_max']
    #   mode = params['rescale_mode']
    #   img, labels = rescale_augment(img, labels, min_f, max_f, mode)

    # # Elastic warp
    # if params['elastic_warp']:
    #   d = params['elastic_d']
    #   n = params['elastic_n']
    #   sigma = params['elastic_sigma']

    #   img, labels = elastic_warp_augment(img, labels, d, n, sigma)
      
    # # Blur
    # if params['blur']:
    #   sigma = params['blur_sigma']
    #   prob = params['blur_prob']
    #   img = blur_augment(img, sigma, prob)

    # # Misalign slip
    # if params['misalign_slip']:
    #   p = params['misalign_slip_prob']
    #   delta = params['misalign_slip_delta']
    #   shift_labels = params['misalign_slip_shift_labels']
    #   img, labels = misalign_slip_augment(img, labels, p, delta, shift_labels)

    # # Misalign translation      
    # if params['misalign_translation']:
    #   p = params['misalign_translation_prob']
    #   delta = params['misalign_translation_delta']
    #   shift_labels = params['misalign_translation_shift_labels']
    #   img, labels = misalign_translation_augment(img, labels, p, delta, shift_labels)

    # # Missing Section
    # if params['missing_section']:
    #   p = params['missing_section_prob']
    #   fill = params['missing_section_fill']
    #   img = missing_section_augment(img, p, fill)


    # # Circle
    # if params['circle']:
    #   p = params['circle_prob']
    #   r = params['circle_radius']
    #   fill = params['circle_fill']
    #   img = circle_augment(img, p, r, fill)

    # if params['grey']:
    #   raise NotImplementedError
    
    # if params['noise']:
    #   sigma = params['noise_sigma']
    #   img = noise_augment(img, sigma)

    # if params['sin']:
    #   a = params['sin_a']
    #   f = params['sin_f']
    #   img = sin_augment(img, a, f)

    # if params['box']:
    #   n = params['box_n']
    #   r = params['box_r']
    #   z = params['box_z']
    #   fill = params['box_fill']
    #   img = box_augment(img, n, r, z, fill)

    # Return
    return sample