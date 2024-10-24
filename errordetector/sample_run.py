# Import necessary packages
import numpy as np
from sys import argv

from cloudvolume import CloudVolume

from train.option import Options
from test.utils import *
from test.model import Model

from utils.utils import *

import torch
from torch.nn import functional as F

from nets.unet_all import *


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

patch_image = vol_image[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])].astype(np.uint8).T
patch_machine_label = vol_machine_label[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])].astype(np.int32).T
patch_human_label = vol_human_label[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])].astype(np.int32).T

patch_size.reverse()
sample_machine_label = torch.reshape(torch.tensor(patch_machine_label), [1]+patch_size)
sample_human_label = torch.reshape(torch.tensor(patch_human_label), [1]+patch_size)

patch_obj_mask = object_mask(patch_machine_label)
sample_obj_mask = torch.reshape(torch.tensor(patch_obj_mask, dtype=torch.float32), [1]+patch_size)


# Inference options
opt = Options()

opt.net = UNet()
opt.model_dir = argv[6] + 'model/'
opt.chkpt_num = int(argv[7])

opt.in_spec = ["obj_mask", "image"]

opt.pretrain = None 
opt.gpu_ids = ["0"]

opt.out_size = (25,16,16)


# Load model
model = load_model(opt)

# Prepare input sample
sample = {}
sample["obj_mask"] = torch.reshape(sample_obj_mask, [1,1] + patch_size)
sample["image"] = torch.tensor(np.reshape(patch_image/255, [1,1] + patch_size), dtype=torch.float32)
error2_truth = torch.reshape(has_error(sample_obj_mask, sample_human_label, (1,)+opt.out_size), opt.out_size)*255
error2_truth = error2_truth.numpy().astype('uint8')


# Discriminate
preds = forward(model, sample)
error2 = (preds["error2"]*255).cpu().detach().numpy()
error2 = np.reshape(error2, error2.shape[2:]).astype('uint8')
print(">>>>> Label : ", np.max(error2_truth)/255)
print(">>>>> Prediction : ", np.max(error2)/255)


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
		encoding = 'raw', # other option: 'jp2eg' but it's lossy
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
upload_cloud("gs://neuroglancer/alex/pinky10/sample_image_590_1195_226", patch_image.T, 'image', 'uint8', [4,4,40], patch_size[::-1])
upload_cloud("gs://neuroglancer/alex/pinky10/sample_obj_mask_590_1195_226", patch_obj_mask.T.astype('uint32'), 'segmentation', 'uint32', [4,4,40], patch_size[::-1])
upload_cloud("gs://neuroglancer/alex/pinky10/sample_human_label_590_1195_226", patch_human_label.T.astype('uint32'), 'segmentation', 'uint32', [4,4,40], patch_size[::-1])
upload_cloud("gs://neuroglancer/alex/pinky10/sample_error_pred_590_1195_226", error2.T, 'image', 'uint8', [64,64,40], [16,16,25])
upload_cloud("gs://neuroglancer/alex/pinky10/sample_error_truth_590_1195_226", error2_truth.T, 'image', 'uint8', [64,64,40], [16,16,25])