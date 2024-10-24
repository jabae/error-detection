from __future__ import print_function
import imp
import os
import time

import torch
import torch.nn.functional as F

from dataset import *

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
                "valid": "valid.h5"
            }
        )

        exp_dir = data_dir + vol_name
        exp_name = 'Inference'

        patch_size = (33,320,320)
        visited_size = (16,160,160)
        chunk_size = (64,512,512)

        # Load test data
        human_labels = TEST.human_labels
        machine_labels = TEST.machine_labels
        valid = TEST.valid[0]
        
        volume_size = human_labels[0].shape[1:]
        padded_volume_size = human_labels[0].shape
        padded_patch_size = (1,) + patch_size
        
        full_labels_lies = MultiVolume(machine_labels, padded_patch_size)
        full_labels_truth = MultiVolume(human_labels, padded_patch_size)


        focus_array = sample_objects_chunked(machine_labels[0].reshape(volume_size),
                    volume_size, patch_size, visited_size, chunk_size, mip=1)
       
        h5write(exp_dir + "samples_new.h5", img=focus_array)
        print(str(i) + " samples saved!")


