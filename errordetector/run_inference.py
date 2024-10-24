import numpy as np
import torch

from cloudvolume import CloudVolume

from errordetector.dataset import *

from errordetector.test.utils import *
from errordetector.test.model import Model

from errordetector.utils.utils import *
from errordetector.utils.utils_cloud import *
from errordetector.utils.sample import *

from errordetector.nets.nets import *

from time import time
import argparse


# Input arguments
parser = argparse.ArgumentParser()

parser.add_argument("--src_image", type=str,
	help='CloudVolume path of source image')
parser.add_argument("--src_seg", type=str,
	help='CloudVolume path of source segmentation')
parser.add_argument("--dst_path", type=str,
	help='Destination path to save errormap')
parser.add_argument("--bbox_start", nargs=3, type=int,
	help='Chunk origin index')
parser.add_argument("--bbox_end", nargs=3, type=int,
	help='Chunk end index')

arg = parser.parse_args()

cv_image = arg.src_image
cv_seg = arg.src_seg
cv_dst = arg.dst_path

bbox_start = arg.bbox_start
bbox_end = arg.bbox_end


# Load data
vol_image = CloudVolume(cv_image, parallel=True, progress=True)
vol_seg = CloudVolume(cv_seg, parallel=True, progress=True)

image = vol_image[bbox_start[0]:bbox_end[0],bbox_start[1]:bbox_end[1],bbox_start[2]:bbox_end[2]][:,:,:,0].T
seg = vol_seg[bbox_start[0]:bbox_end[0],bbox_start[1]:bbox_end[1],bbox_start[2]:bbox_end[2]][:,:,:,0].T.astype('int32')


# Load model
net = UNetMip1()
model_dir = "/usr/people/jabae/seungmount/research/Alex/error_detection/exp/reconstruction_norm_lr_0001/exp_reconstruction_norm_loc_mip1_320_0313/model/"
chkpt_num = 390000

arg.net = net
arg.model_dir = model_dir
arg.chkpt_num = chkpt_num
arg.mip = 1
model = load_model(arg)


# Parameters
patch_size = (33,320,320)
volume_size = image.shape
visited_size = (16,160,160)
chunk_size = (64,512,512)

padded_patch_size = (1,1,) + patch_size
padded_volume_size = (1,1,) + volume_size


# Volumes
error_map = np.zeros(volume_size)

vol_image = Volume(image/255, patch_size)
vol_seg = Volume(seg, patch_size)
vol_errormap = Volume(error_map, patch_size)

t0 = time()
focus_list = sample_objects_chunked(vol_seg, volume_size, patch_size, visited_size, chunk_size, mip=2)
# focus_list = sample_objects(vol_seg, volume_size, patch_size, visited_size)

print(">>>>> Error detection...")
for i in range(focus_list.shape[0]):

	focus = focus_list[i,:]

	sample_image = vol_image[focus]
	sample_seg = vol_seg[focus]
	sample_obj_mask = object_mask(sample_seg)

	sample = {}
	sample["image"] = torch.from_numpy(np.reshape(sample_image, padded_patch_size)).float()
	sample["obj_mask"] = torch.from_numpy(np.reshape(sample_obj_mask, padded_patch_size)).float()

	pred = forward(model, sample)

	error = pred["error"]
	error_upsample = F.interpolate(error, scale_factor=(1,16,16), mode="nearest").cpu().detach().numpy()
	error_upsample = np.reshape(error_upsample, patch_size)
	vol_errormap[focus] = np.maximum(vol_errormap[focus], error_upsample*sample_obj_mask)

  # Print iteration stat
	if (i+1) % 100 == 0 or (i+1) <= 10:
	  print("{} / {} done.".format(i+1, focus_list.shape[0]))

elapsed = np.round(time() - t0, 3)

errormap = (vol_errormap.A*255).astype('uint8')
errormap = np.reshape(errormap, volume_size).T
print(">>>>> Detection complete!")
print("Elapsed time = {}".format(elapsed))

upload_cloud(cv_dst, errormap, 'image', 'uint8', [4,4,40], volume_size[::-1])