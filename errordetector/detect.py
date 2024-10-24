#!/usr/bin/env python3
"""
Error detection inference
"""

# Import necessary packages
import numpy as np

import torch
import torch.nn.functional

from errordetector.test.model import Model
from errordetector.test.utils import *
from errordetector.utils.utils import *

from time import time

from errordetector.nets.unet_mip1 import *


def detect(seg, img, opt):

	# Load model
	opt.net = UNet()
	model = load_model(opt)

	# Size parameter
	volume_size = seg.shape
	padded_volume_size = (1,) + volume_size
	patch_size = tuple(opt.patch_shape[::-1])
	padded_patch_size = (1,) + patch_size
	out_size = tuple(opt.out_shape[::-1])

	# Volumes
	if np.max(img) > 10:
		img = img/255
	img = np.reshape(img, padded_volume_size)
	seg = np.reshape(seg.astype('int32'), padded_volume_size)
	visited = np.zeros(padded_volume_size)
	errormap = np.zeros(padded_volume_size)

	full_image = Volume(img, padded_patch_size)
	full_seg = Volume(seg, padded_patch_size)
	full_visited = Volume(visited, (1,16,160,160))
	full_errormap = Volume(errormap, padded_patch_size)

	# Mark boundaries and edge areas visited
	full_visited.A[0,:patch_size[0]//2,:,:] = 1
	full_visited.A[0,:,:patch_size[1]//2,:] = 1
	full_visited.A[0,:,:,:patch_size[2]//2] = 1
	full_visited.A[0,volume_size[0]-patch_size[0]//2:,:,:] = 1
	full_visited.A[0,:,volume_size[1]-patch_size[1]//2:,:] = 1
	full_visited.A[0,:,:,volume_size[2]-patch_size[2]//2:] = 1

	boundary_idx = np.where(full_seg.A==0)
	full_visited.A[boundary_idx] = 1

    # Detect
	t0 = time()
	i = 0
	coverage = 0
	while coverage < 1:

		focus_np = random_coord_valid(volume_size, patch_size)
		focus = torch.tensor(focus_np)
		focus = torch.cat((torch.zeros((1,), dtype=focus.dtype), torch.reshape(focus, (3,))), 0)

        # If the sampled point of visited array is marked, skip.
		if full_visited.A[0,focus[1],focus[2],focus[3]] >= 1:
			continue

        # Prepare input    
		sample_seg = full_seg[focus]
		sample_obj_mask = object_mask(sample_seg)
		sample_image = torch.tensor(full_image[focus], dtype=torch.float32)

		sample = {}
		sample["obj_mask"] = torch.reshape(sample_obj_mask, (1,)+padded_patch_size)
		sample["image"] = torch.reshape(sample_image, (1,)+padded_patch_size)

        # Discriminate
		preds = forward(model, sample)

        # Update error map
		sample_errormap = preds["error2"]
		sample_errormap_us = F.upsample(torch.reshape(sample_errormap, (1,)+out_size),
        								scale_factor=16, mode='nearest').cpu().detach()
		
		full_errormap[focus] = np.maximum(full_errormap[focus], sample_errormap_us*sample["obj_mask"])

		full_visited[focus] = torch.tensor(full_visited[focus], dtype=torch.float) + sample_obj_mask[:,8:24,80:240,80:240]
		
		# Stats        
		coverage = np.round(np.sum(full_visited.A>=1)/np.prod(volume_size),4)
		elapsed = np.round(time() - t0,3)

		i = i + 1

		if i % 100 == 0 or i <=10:
			print("Iter:  " + str(i) + ", elapsed time = " + str(elapsed) + ", coverage = " + str(coverage))


	return np.reshape((full_errormap.A*255).astype('uint8'), volume_size)