from __future__ import print_function
import imp
import os
import time

import torch
import torch.nn.functional as F

from errordetector.test.model import Model
from errordetector.test.utils import *

from errordetector.utils.utils import *

from dataset import *


# Test sample generation
if __name__ == "__main__":

    data_dir = "/usr/people/jabae/seungmount/Omni/TracerTasks/pinky/proofreading/"

    # Volumes
    vol_list = ["chunk_14977-17024_27265-29312_4003-4258.omni.files/",
                "chunk_14977-17024_28801-30848_4003-4258.omni.files/",
                "chunk_14977-17024_30337-32384_4003-4258.omni.files/",
                "chunk_16513-18560_27265-29312_4003-4258.omni.files/",
                "chunk_16513-18560_28801-30848_4003-4258.omni.files/",
                "chunk_16513-18560_30337-32384_4003-4258.omni.files/",
                "chunk_18049-20096_27265-29312_4003-4258.omni.files/",
                "chunk_18049-20096_28801-30848_4003-4258.omni.files/",
		"chunk_18049-20096_30337-32384_4003-4258.omni.files/"]
    
    for i in range(len(vol_list)):

        vol_name = vol_list[i]

        TEST = MultiDataset(
            [
                os.path.expanduser(data_dir + vol_name)
            ],
            {
                "machine_labels": "lzf_mean_agg_tr.h5",
                "human_labels": "lzf_proofread.h5",
                "samples": "samples_new.h5",
            }
        )

        exp_dir = data_dir + vol_name

        data = TEST
        patch_size = (33,320,320)
        out_size = (33,20,20)

        # Load test data
        human_labels = TEST.human_labels
        machine_labels = TEST.machine_labels
        samples = TEST.samples

        volume_size = human_labels[0].shape[1:]
        padded_volume_size = human_labels[0].shape
        padded_patch_size = (1,) + patch_size
        padded_out_size = (1,) + out_size

        errormap = [np.zeros(padded_volume_size)]

        full_labels_lies = MultiVolume(machine_labels, padded_patch_size)
        full_labels_truth = MultiVolume(human_labels, padded_patch_size)
        full_samples = MultiVolume(samples, (1,3), indexing='CORNER')
        full_errormap = MultiVolume(errormap, padded_patch_size)

        print(samples[0].shape)
        for i in range(samples[0].shape[0]):

            focus = full_samples[0, (i,0)]
            focus = torch.cat((torch.zeros((1,), dtype=focus.dtype), torch.reshape(focus, (3,))), 0).numpy()
            
            # Filter samples
            sample_machine_label = full_labels_lies[0, focus]
            sample_human_label = full_labels_truth[0, focus]
            sample_obj_mask = object_mask(sample_machine_label)
            
            error = has_error(sample_obj_mask, sample_human_label, padded_out_size)
            error_upsample = F.upsample(error, scale_factor=16, mode='nearest').cpu().detach()
            full_errormap[0, focus] = np.maximum(full_errormap[0, focus], error_upsample*sample_obj_mask)

            print(i+1,'/',samples[0].shape[0])
            
        full_errormap.As[0].A = (full_errormap.As[0].A*255).astype('uint8')
        h5write(exp_dir + "gt_errormap.h5", img=np.reshape(full_errormap.As[0].A, volume_size).T)
        print(str(i+1) + " samples saved!")


