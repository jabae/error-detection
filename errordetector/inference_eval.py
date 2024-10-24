from __future__ import print_function
import imp
import os
import time

import torch
import torch.nn.functional as F

from dataset import *

from test.model import Model
from test.utils import *
from test.option import Options

from nets.unet_mip1 import *

from utils.utils import *

from sys import argv


def test(opt):
    # Load test data
    patch_size = opt.patch_size
    padded_patch_size = (1,) + patch_size
    
    full_labels_lies = MultiVolume(opt.data.machine_labels, padded_patch_size)
    full_errormap_lies = MultiVolume(opt.data.pred_errormap, padded_patch_size)
    full_samples = MultiVolume(opt.data.samples, (1,3), indexing='CORNER')

    error_pred = np.array([])
    t0 = time.time()
    for i in range(opt.data.samples[0].shape[0]):
        
        t1 = time.time()

        focus_raw = full_samples[0, (i,0)]
        focus = torch.cat((torch.zeros((1,), dtype=focus_raw.dtype), torch.reshape(focus_raw, (3,))), 0)
        
        sample_machine_label = full_labels_lies[0, focus]
        sample_obj_mask = object_mask(sample_machine_label)

        sample_pred_err = full_errormap_lies[0, focus]
        pred0 = np.reshape(torch.max(torch.tensor(sample_pred_err[0,14:18,140:180,140:180],dtype=torch.float)*torch.tensor(sample_obj_mask[0,14:18,140:180,140:180],dtype=torch.float)).numpy(),(-1,))
        # pred0 = np.reshape(sample_pred_err[0,16,160,160].numpy(),(-1,))
        error_pred = np.concatenate((error_pred,pred0))

        # Stats
        elapsed = np.round(time.time() - t1, 3)

        # Print iteration stat
        if i % 100 == 0 or i <= 10:
            print("Iter:  " + str(i) + ", elapsed time = " + str(elapsed))


    # Save final files.
    h5write(opt.output_dir + opt.exp_name + "_pred.h5", img=error_pred)
    print("Elapsed = " + str(np.round(time.time()-t0, 3)))
    

# Run inference
if __name__ == "__main__":

    # Test volume
    TEST = MultiDataset(
        {
            "machine_labels": "/usr/people/jabae/seungmount/Omni/TracerTasks/pinky/proofreading/chunk_18049-20096_30337-32384_4003-4258.omni.files/lzf_mean_agg_tr.h5",
            "pred_errormap": argv[1],
            "samples": argv[2]
        }
    )
    
    # Options
    opt = Options()
    
    opt.output_dir = '/usr/people/jabae/seungmount/research/Alex/errordetection/test_vol/'
    opt.exp_name = 'edX_1217'

    opt.data = TEST
    opt.patch_size = (33,320,320)

    test(opt)