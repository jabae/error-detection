# Import necessary packages
import numpy as np
from sys import argv

from cloudvolume import CloudVolume

from deepem.train.option import Options
from test.utils import *

import torch
from torch.nn import functional as F

from unet import *


# Arguments
file_image = argv[1] # Directory for EM image layer
file_machine_label = argv[2] # Directory for machine label layer
file_human_label = argv[3] # Directory for human label layer
focus = [int(argv[4][1:-1].split(',')[i]) for i in range(3)] # [x,y,z] without any spaces 
patch_size = [int(argv[5][1:-1].split(',')[i]) for i in range(3)] # [x,y,z] without any spaces


# Export volume from cloudvolume
vol_image = CloudVolume(file_image, parallel=True, progress=True)
vol_machine_label = CloudVolume(file_machine_label, parallel=True, progress=True)
vol_human_label = CloudVolume(file_human_label, parallel=True, progress=True)

# Extract sample patch from the volume
corner = np.array(focus) - np.array([x/2 for x in patch_size], dtype=np.int32)
corner = np.reshape(corner,(-1,))

patch_image = vol_image[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])]
patch_machine_label = vol_machine_label[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])]
patch_human_label = vol_human_label[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])]

lies_id = vol_machine_label[focus]
patch_obj_mask = np.array(patch_machine_label == lies_id, dtype=np.uint32)
truth_id = vol_human_label[focus]
patch_obj_label = np.array(patch_human_label == truth_id, dtype=np.uint32)

# Load model
opt = Options()

opt.model_dir = argv[6] + 'model/'
opt.chkpt_num = int(argv[7])

opt.in_spec = ["obj_mask", "image"]
opt.out_spec = ["error0", "error1", "error2"]

opt.pretrain = None 
opt.gpu_ids = ["0"]


# model = load_model(opt)
net = UNet()
model = Model(net, opt)

if opt.chkpt_num > 0:
    model = load_chkpt(model, opt.model_dir, opt.chkpt_num)

# Prepare input sample
patch_size.reverse()

inputs_list = []
inputs_list.append(np.reshape(patch_obj_mask.T, [1,1] + patch_size).astype(np.int32))
inputs_list.append(np.reshape(patch_image.T, [1,1] + patch_size).astype(np.int32))

# 2-channel with machine generated object mask and raw EM image
inputs = np.concatenate(inputs_list, axis=1)
inputs = torch.tensor(inputs, dtype=torch.float32)


# Discriminate
preds = forward(model, inputs)

error0 = torch.round(F.sigmoid(preds["error0"])).detach().numpy().astype('uint8')
error1 = torch.round(F.sigmoid(preds["error1"])).detach().numpy()
error1 = np.reshape(error1, error1.shape[2:]).astype('uint8')
error2 = torch.round(F.sigmoid(preds["error2"])).detach().numpy()
error2 = np.reshape(error2, error2.shape[2:]).astype('uint8')

print(">>>>> Error0 : ", error0[0,0,0,0,0])


# Upload volume to cloud
def upload_cloud(cloud_dir, volume, layer_type, dtype, resolution, volume_size):
	'''
	cloud_dir : Cloud directory to upload
	vol : Volume to upload
	'''
	info = CloudVolume.create_new_info(
		num_channels = 1,
		layer_type = layer_type, # 'image' or 'segmentation'
		data_type = dtype, # can pick any popular uint
		encoding = 'raw', # other option: 'jpeg' but it's lossy
		resolution = resolution, # X,Y,Z values in nanometers
		voxel_offset = [ 0, 0, 0 ], # values X,Y,Z values in voxels
		chunk_size = [ 128, 128, 64 ], # rechunk of image X,Y,Z in voxels
		volume_size = volume_size, # X,Y,Z size in voxels
	)


	vol = CloudVolume(cloud_dir, parallel=True, progress=True, cdn_cache=False, info=info)
	vol.provenance.description = "Pinky10 volume for error detection"
	vol.provenance.owners = ['jabae@princeton.edu'] # list of contact email addresses

	vol.commit_info() # generates gs://bucket/dataset/layer/info json file
	vol.commit_provenance() # generates gs://bucket/dataset/layer/provenance json file

	vol[:,:,:] = volume


# Upload
upload_cloud("gs://neuroglancer/alex/pinky10/sample_image", np.reshape(patch_image, patch_image.shape[:3]), 'image', 'uint8', [4,4,40], [256,256,25])
upload_cloud("gs://neuroglancer/alex/pinky10/sample_obj_mask", np.reshape(patch_obj_mask, patch_obj_mask.shape[:3]), 'segmentation', 'uint32', [4,4,40], [256,256,25])
upload_cloud("gs://neuroglancer/alex/pinky10/sample_obj_label", np.reshape(patch_obj_label, patch_obj_label.shape[:3]), 'segmentation', 'uint32', [4,4,40], [256,256,25])
upload_cloud("gs://neuroglancer/alex/pinky10/error_map", 1 - error2.T, 'segmentation', 'uint8', [64,64,40], [16,16,25])


