# Import necessary packages
import numpy as np
from sys import argv

from cloudvolume import CloudVolume

from dataset import *
from utils.utils import *
from utils.utils_cloud import *


samples = h5read('/usr/people/jabae/seungmount/research/Alex/errordetection/test_vol/valid_test_samples_raw.h5')
p = h5read('/usr/people/jabae/seungmount/research/Alex/errordetection/test_vol/pred_valid_test_samples_raw_center.h5')
l = h5read('/usr/people/jabae/seungmount/research/Alex/errordetection/test_vol/label_valid_test_samples_raw.h5')

idx = int(argv[1])

file_image = "gs://neuroglancer/alex/pinky10/volume_3_3/image_3_3_1" # Directory for EM image layer
file_machine_label = "gs://neuroglancer/kisuk/pinky/proofreading/chunk_3_3/machine_label" # Directory for machine label layer
file_human_label = "gs://neuroglancer/kisuk/pinky/proofreading/chunk_3_3/seg" # Directory for human label layer
file_errormap_truth = "gs://neuroglancer/alex/pinky10/volume_3_3/groundtruth_errormap_noglia"
file_errormap_pred = "gs://neuroglancer/alex/pinky10/volume_3_3/pred_errormap"
focus = samples[idx,:][::-1] # [x,y,z] without any spaces 
patch_size = [320,320,33] # [x,y,z] without any spaces


# Export volume from cloudvolume
vol_image = CloudVolume(file_image, parallel=True, progress=True)
vol_machine_label = CloudVolume(file_machine_label, parallel=True, progress=True)
vol_human_label = CloudVolume(file_human_label, parallel=True, progress=True)
vol_errormap_truth = CloudVolume(file_errormap_truth, parallel=True, progress=True)
vol_errormap_pred = CloudVolume(file_errormap_pred, parallel=True, progress=True)


# Extract sample patch from the volume
corner = np.array(focus) - np.array([x/2 for x in patch_size], dtype=np.int32)
corner = np.reshape(corner,(-1,))

patch_image = vol_image[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])]
patch_machine_label = vol_machine_label[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])]
patch_human_label = vol_human_label[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])]
patch_errormap_truth = vol_errormap_truth[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])]
patch_errormap_pred = vol_errormap_pred[tuple([slice(corner[i],corner[i]+patch_size[i]) for i in range(len(patch_size))])]

patch_obj_mask = object_mask(patch_machine_label).astype('uint8')

# Upload
upload_cloud("gs://neuroglancer/alex/pinky10/sample/sample_image", patch_image, 'image', 'uint8', [4,4,40], patch_size)
upload_cloud("gs://neuroglancer/alex/pinky10/sample/sample_machine_label", patch_obj_mask, 'segmentation', 'uint8', [4,4,40], patch_size)
upload_cloud("gs://neuroglancer/alex/pinky10/sample/sample_human_label", patch_human_label, 'segmentation', 'uint32', [4,4,40], patch_size)
upload_cloud("gs://neuroglancer/alex/pinky10/sample/sample_error_pred", patch_errormap_pred, 'image', 'uint8', [4,4,40], patch_size)
upload_cloud("gs://neuroglancer/alex/pinky10/sample/sample_error_truth", patch_errormap_truth, 'image', 'uint8', [4,4,40], patch_size)


print("Label = " + str(l[idx]) + ", " + "Pred = " + str(p[idx]))