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
    human_labels = opt.data.human_labels
    machine_labels = opt.data.machine_labels
    errormap_truth = opt.data.truth_errormap
    errormap_lies = opt.data.pred_errormap
    valid = TEST.valid[0]

    volume_size = machine_labels[0].shape[1:]
    padded_volume_size = machine_labels[0].shape
    patch_size = opt.patch_size
    padded_patch_size = (1,) + patch_size

    visited = [np.zeros(padded_volume_size)]
    
    full_labels_lies = MultiVolume(machine_labels, padded_patch_size)
    full_labels_truth = MultiVolume(human_labels, padded_patch_size)
    full_errormap_truth = MultiVolume(errormap_truth, padded_patch_size)
    full_errormap_lies = MultiVolume(errormap_lies, padded_patch_size)
    full_visited = MultiVolume(visited, (1,16,160,160))

    samples = opt.data.samples
    full_samples = MultiVolume(samples, (1,3), indexing='CORNER')

    # Mark boundaries and edge areas visited
    full_visited.As[0].A[0,:patch_size[0]//2,:,:] = 1
    full_visited.As[0].A[0,:,:patch_size[1]//2,:] = 1
    full_visited.As[0].A[0,:,:,:patch_size[2]//2] = 1
    full_visited.As[0].A[0,volume_size[0]-patch_size[0]//2:,:,:] = 1
    full_visited.As[0].A[0,:,volume_size[1]-patch_size[1]//2:,:] = 1
    full_visited.As[0].A[0,:,:,volume_size[2]-patch_size[2]//2:] = 1

    idx = np.where(full_labels_lies.As[0].A==0)
    full_visited.As[0].A[idx] = 1

    focus_list = np.array([])
    error_label = np.array([])
    error_pred = np.array([])
    coverage = 0
    t0 = time.time()
    for i in range(opt.data.samples[0].shape[0]):
        
        t1 = time.time()

        focus_raw = full_samples[0, (i,0)]
        focus = torch.cat((torch.zeros((1,), dtype=focus_raw.dtype), torch.reshape(focus_raw, (3,))), 0)
        

        # Filter samples
        sample_machine_label = full_labels_lies[0, focus]
        sample_human_label = full_labels_truth[0, focus]
        sample_obj_mask = object_mask(sample_machine_label)

        sample_truth_err = full_errormap_truth[0, focus]
        sample_pred_err = full_errormap_lies[0, focus]

        # error_inner = has_error(sample_obj_mask[:,14:18,140:180,140:180], sample_human_label[:,14:18,140:180,140:180],(1,1,1,1))
        # error_outer = has_error(sample_obj_mask[:,12:20,120:200,120:200], sample_human_label[:,12:20,120:200,120:200],(1,1,1,1))

        # error0 = np.reshape(error_inner.numpy(),(-1,))
        # error_label = np.concatenate((error_label,error0))
        
        # pred0 = np.reshape(torch.max(torch.tensor(sample_pred_err[0,14:18,140:180,140:180],dtype=torch.uint8)*torch.tensor(sample_obj_mask[0,14:18,140:180,140:180],dtype=torch.uint8)).numpy(),(-1,))
        pred0 = np.reshape(sample_pred_err[0,16,160,160].numpy(),(-1,))
        error_pred = np.concatenate((error_pred,pred0))

        # Stats
        elapsed = np.round(time.time() - t0, 3)

        # Print iteration stat
        if i % 100 == 0 or i <= 10:
            print("Iter:  " + str(i) + ", elapsed time = " + str(elapsed))


    # Save final files.
    # h5write(opt.output_dir + opt.exp_name + "_label.h5", img=error_label)
    h5write(opt.output_dir + opt.exp_name + "_pred.h5", img=error_pred)
    print("Elapsed = " + str(np.round(time.time()-t0,3)))
    


# Run inference
if __name__ == "__main__":

    data_dir = "/usr/people/jabae/seungmount/Omni/TracerTasks/pinky/proofreading/"

    # Test volume
    TEST = MultiDataset(
        [
            os.path.expanduser(data_dir + "chunk_18049-20096_30337-32384_4003-4258.omni.files/")
        ],
        {
            "machine_labels": "lzf_mean_agg_tr.h5",
            "human_labels": "lzf_proofread.h5",
            "valid": "valid.h5"
        }
    )

    truth_errormap = h5read(argv[1])
    pred_errormap = h5read(argv[2])
    TEST.truth_errormap = [np.reshape(truth_errormap, (1,)+truth_errormap.shape)]
    TEST.pred_errormap = [np.reshape(pred_errormap, (1,)+pred_errormap.shape)]

    samples = h5read('/usr/people/jabae/seungmount/research/Alex/errordetection/test_vol/valid_test_samples_errormap.h5')
    TEST.samples = [samples]
    
    # Options
    opt = Options()
    
    opt.output_dir = '/usr/people/jabae/seungmount/research/Alex/errordetection/test_vol/'
    opt.exp_name = 'edX_1205'

    opt.data = TEST
    opt.patch_size = (33,320,320)

    test(opt)