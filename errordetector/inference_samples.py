from __future__ import print_function
import imp
import os
import time

import torch
import torch.nn.functional as F

from errordetector.test.model import Model
from errordetector.test.utils import *

from errordetector.utils.utils import *

from errordetector.nets.nets import *

from dataset import *


# Test sample generation
if __name__ == "__main__":

    net = UNetMip1()
    model_dir = '/usr/people/jabae/seungmount/research/Alex/error_detection/exp/reconstruction_norm_lr_0001/exp_reconstruction_norm_loc_mip1_320_0313/model/'
    chkpt_num = 360000

    model = load_model(net, model_dir, chkpt_num, 1)

    data_dir = "/usr/people/jabae/seungmount/Omni/TracerTasks/pinky/proofreading/"

    # Test volume
    TEST = MultiDataset(
        [
            os.path.expanduser(data_dir + "chunk_18049-20096_30337-32384_4003-4258.omni.files/")
        ],
        {
            "image": "image.h5",
            "machine_labels": "lzf_mean_agg_tr.h5",
            "human_labels": "lzf_proofread.h5",
            "samples": "valid_test_samples_raw.h5",
        }
    )

    exp_dir = '/usr/people/jabae/seungmount/research/Alex/error_detection/pinky10/test_vol/'

    data = TEST
    patch_size = (33,320,320)
    out_size = (33,20,20)

    # Load test data
    image = TEST.image
    human_labels = TEST.human_labels
    machine_labels = TEST.machine_labels
    samples = TEST.samples

    volume_size = human_labels[0].shape[1:]
    padded_volume_size = human_labels[0].shape
    padded_patch_size = (1,) + patch_size
    padded_out_size = (1,) + out_size

    errormap = [np.zeros(padded_volume_size)]

    full_image = MultiVolume(image, padded_patch_size)
    full_labels_lies = MultiVolume(machine_labels, padded_patch_size)
    full_labels_truth = MultiVolume(human_labels, padded_patch_size)
    full_samples = MultiVolume(samples, (1,3), indexing='CORNER')
    full_errormap = MultiVolume(errormap, padded_patch_size) 

    for i in range(samples[0].shape[0]):

        focus = full_samples[0, (i,0)]
        
        # Filter samples
        sample_machine_label = full_labels_lies[0, focus]
        sample_human_label = full_labels_truth[0, focus]
        sample_obj_mask = object_mask(sample_machine_label)
        sample_image = full_image[0, focus]
        
        sample = {}
        sample["obj_mask"] = torch.reshape(sample_obj_mask, (1,)+sample_obj_mask.shape)
        sample["image"] = torch.reshape(sample_image, (1,)+sample_image.shape)

        pred = forward(model, sample)
        error = pred["error2"]
        error_upsample = F.upsample(error, scale_factor=(1,16,16), mode='nearest').cpu().detach()
        error_loc = error_upsample*sample_obj_mask
        full_errormap[0, focus] = np.maximum(full_errormap[0, focus], error_loc)

        print(i+1,'/',samples[0].shape[0])
        
    full_errormap.As[0].A = (full_errormap.As[0].A*255).astype('uint8')
    h5write(exp_dir + "errormap_v1_360k_0520.h5", img=np.reshape(full_errormap.As[0].A, volume_size))
    print(str(i+1) + " samples saved!")


