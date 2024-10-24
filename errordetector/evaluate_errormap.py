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
    i = 0
    c = 0  
    coverage = 0
    t0 = time.time()
    while coverage < 1:
        c = c + 1
        
        t1 = time.time()

        focus_np = random_coord_valid(volume_size, patch_size)
        focus = torch.tensor(focus_np)
        focus = torch.cat((torch.zeros((1,), dtype=focus.dtype), torch.reshape(focus, (3,))), 0)
        
        # If the sampled point of visited array is marked, skip.
        if full_visited.As[0].A[0,focus[1],focus[2],focus[3]] >= 1:
            continue

        # Filter samples
        sample_machine_label = full_labels_lies[0, focus]
        sample_human_label = full_labels_truth[0, focus]
        obj_id = sample_human_label[0,padded_patch_size[1]//2,padded_patch_size[2]//2,padded_patch_size[3]//2]
        sample_obj_mask = object_mask(sample_machine_label)

        if valid[obj_id] == 0:
            full_visited[0, focus] = torch.tensor(full_visited[0, focus], dtype=torch.float) + sample_obj_mask[:,8:24,80:240,80:240]
            continue
        
        sample_truth_err = full_errormap_truth[0, focus]
        sample_pred_err = full_errormap_lies[0, focus]

        error_inner = has_error(sample_obj_mask[:,14:18,140:180,140:180], sample_human_label[:,14:18,140:180,140:180],(1,1,1,1))
        error_outer = has_error(sample_obj_mask[:,12:20,120:200,120:200], sample_human_label[:,12:20,120:200,120:200],(1,1,1,1))

        # error_inner = torch.max(torch.tensor(sample_truth_err[:,14:18,140:180,140:180],dtype=torch.uint8)*torch.tensor(sample_obj_mask[0,14:18,140:180,140:180],dtype=torch.uint8))
        # error_outer = torch.max(torch.tensor(sample_truth_err[:,12:20,120:200,120:200],dtype=torch.uint8)*torch.tensor(sample_obj_mask[0,12:20,120:200,120:200],dtype=torch.uint8))

        if error_inner == 0 and error_outer == 1:
            print('Invalid sample.')
            continue

        error0 = np.reshape(error_inner.numpy(),(-1,))
        error_label = np.concatenate((error_label,error0))
        
        
        # pred0 = np.reshape(torch.max(torch.tensor(sample_pred_err[0,14:18,140:180,140:180],dtype=torch.uint8)*torch.tensor(sample_obj_mask[0,14:18,140:180,140:180],dtype=torch.uint8)).numpy(),(-1,))
        pred0 = np.reshape(sample_pred_err[0,16,160,160].numpy(),(-1,))
        error_pred = np.concatenate((error_pred,pred0))

        # Add sampled point in the list
        focus_list = np.concatenate((focus_list, np.reshape(focus_np,(-1,))))

        # Mark visited segment
        full_visited[0, focus] = torch.tensor(full_visited[0, focus], dtype=torch.float) + sample_obj_mask[:,8:24,80:240,80:240]  
        
        # Stats
        coverage = np.round(np.sum(full_visited.As[0].A>=1)/np.prod(volume_size),4)
        elapsed = np.round(time.time() - t0, 3)

        i = i + 1

        # Print iteration stat
        if i % 100 == 0 or i <= 10:
            print("Iter:  " + str(c) + ", elapsed time = " + str(elapsed) + ", coverage = " + str(coverage))


    # Save final files.
    focus_array = np.reshape(focus_list, (-1,3)).astype('uint32')
    error_label = (error_label/255.0).astype('float32')
    error_pred = (error_pred/255.0).astype('float32')

    h5write(opt.output_dir + opt.exp_name + "_test_samples.h5", img=focus_array)
    h5write(opt.output_dir + opt.exp_name + "_label.h5", img=error_label)
    h5write(opt.output_dir + opt.exp_name + "_pred.h5", img=error_pred)
    print("Elapsed = " + str(np.round(time.time()-t0,3)))
    print(str(i+1) + " samples saved!")
    


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

    # Options
    opt = Options()
    
    opt.output_dir = '/usr/people/jabae/seungmount/research/Alex/errordetection/test_vol/'
    opt.exp_name = 'edX_1205'

    opt.data = TEST
    opt.patch_size = (33,320,320)

    test(opt)