import numpy as np
import torch

from cloudvolume import CloudVolume

from errordetector.dataset import *

from errordetector.test.utils import *
from errordetector.test.model import Model

from errordetector.utils.utils import *
from errordetector.utils.utils_cloud import *

from errordetector.nets.nets import *

import argparse


# Input arguments
parser = argparse.ArgumentParser()

parser.add_argument("sample_loc", nargs=3, type=int)

arg = parser.parse_args()
focus = arg.sample_loc[::-1]


# Load Data
cv_image = "gs://neuroglancer/alex/pinky10/volume_3_3/image_3_3_1"
cv_machine_label = "gs://neuroglancer/alex/pinky10/volume_3_3/machine_label_3_3_1"
cv_human_label = "gs://neuroglancer/alex/pinky10/volume_3_3/human_label_3_3_1"

vol_image = CloudVolume(cv_image, parallel=True, progress=False)
vol_machine_label = CloudVolume(cv_machine_label, parallel=True, progress=False)
vol_human_label = CloudVolume(cv_human_label, parallel=True, progress=False)


# Extract chunk
patch_size = (320,320,33)

corner = np.array(focus) - np.array([x//2 for x in patch_size], dtype=np.int32)
corner = np.reshape(corner,(-1,))

patch_image = vol_image[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])].T
patch_machine_label = vol_machine_label[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])].T
patch_human_label = vol_human_label[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])].T
patch_obj_mask = object_mask(patch_machine_label)

sample = {}
sample["obj_mask"] = torch.reshape(torch.from_numpy(patch_obj_mask), (1,1)+patch_size[::-1]).float()
sample["image"] = torch.reshape(torch.from_numpy(patch_image/255), (1,1)+patch_size[::-1]).float()


# Discriminate
net = UNetMip1()
model_dir = "/usr/people/jabae/seungmount/research/Alex/error_detection/exp/reconstruction_norm_lr_0001/exp_reconstruction_norm_loc_mip1_320_0313/model/"
chkpt_num = 390000

model = load_model(net, model_dir, chkpt_num, 1)

pred = forward(model, sample)
error = pred["error2"]
error_upsample = F.upsample(error, scale_factor=(1,16,16), mode='nearest').cpu().detach().numpy()
error_loc = error_upsample*patch_obj_mask


# Upload
img_upload = patch_image.T
obj_upload = (patch_obj_mask.T*255).astype('uint8')
seg_upload = patch_human_label.T
error_upload = np.reshape(error_loc*255, patch_size[::-1]).astype('uint8').T

upload_cloud("gs://neuroglancer/alex/pinky10/sample/image", img_upload, 'image', 'uint8', [4,4,40], patch_size, corner)
upload_cloud("gs://neuroglancer/alex/pinky10/sample/obj_mask", obj_upload, 'image', 'uint8', [4,4,40], patch_size, corner)
upload_cloud("gs://neuroglancer/alex/pinky10/sample/human_label", seg_upload, 'segmentation', 'uint32', [4,4,40], patch_size, corner)
upload_cloud("gs://neuroglancer/alex/pinky10/sample/errormap", error_upload, 'image', 'uint8', [4,4,40], patch_size, corner)
