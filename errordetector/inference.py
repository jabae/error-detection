from __future__ import print_function
import imp
import os
import time

import torch
import torch.nn.functional as F

from dataset import *

from errordetector.test.model import Model
from errordetector.test.utils import *
from errordetector.test.option import Options

from errordetector.nets.unet_mip1 import *

from errordetector.utils.utils import *

from sys import argv

def test(opt):

    # Load model
    model = load_model(opt)

    # Load test data
    machine_labels = opt.data.machine_labels
    image = opt.data.image
    
    volume_size = machine_labels[0].shape[1:]
    padded_volume_size = machine_labels[0].shape
    patch_size = opt.patch_size
    padded_patch_size = (1,) + patch_size

    visited = [np.zeros(padded_volume_size)]
    error_map = [np.zeros(padded_volume_size)]
    
    full_image = MultiVolume(image, padded_patch_size)
    full_labels_lies = MultiVolume(machine_labels, padded_patch_size)
    full_visited = MultiVolume(visited, (1,16,160,160))
    full_errormap_lies = MultiVolume(error_map, padded_patch_size)

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
    i = 0
    coverage = 0
    # Discriminate test samples
    while coverage < 1:

        t0 = time.time()

        focus_np = random_coord_valid(volume_size, patch_size)
        focus = torch.tensor(focus_np)
        focus = torch.cat((torch.zeros((1,), dtype=focus.dtype), torch.reshape(focus, (3,))), 0)
        
        # If the sampled point of visited array is marked, skip.
        if full_visited.As[0].A[0,focus[1],focus[2],focus[3]] >= 1:
            continue
        
        sample_machine_label = full_labels_lies[0, focus]
        sample_obj_mask = object_mask(sample_machine_label)    
        
        # Add sampled point in the list
        focus_list = np.concatenate((focus_list, np.reshape(focus_np,(-1,))))

        # Prepare sample
        sample_image = full_image[0, focus]
        
        sample = {}
        sample["obj_mask"] = torch.reshape(sample_obj_mask, (1,)+sample_obj_mask.shape)
        sample["image"] = torch.reshape(sample_image, (1,)+sample_image.shape)
        
        # Discriminate
        preds = forward(model, sample)

        # Predictions
        pred2 = preds["error2"]
        
        pred2_upsample = F.upsample(torch.reshape(pred2, (1,)+opt.out_size), scale_factor=16, mode='nearest').cpu().detach().numpy()
        full_errormap_lies[0, focus] = np.maximum(full_errormap_lies[0, focus], pred2_upsample*sample["obj_mask"])

        # Mark visited segment
        full_visited[0, focus] = torch.tensor(full_visited[0, focus], dtype=torch.float) + sample_obj_mask[:,8:24,80:240,80:240]  
        
        # Stats
        coverage = np.round(np.sum(full_visited.As[0].A>=1)/np.prod(volume_size),4)
        elapsed = np.round(time.time() - t0, 3)

        i = i + 1

        # Print iteration stat
        if i % 100 == 0 or i <= 10:
            print("Iter:  " + str(i) + ", elapsed time = " + str(elapsed) + ", coverage = " + str(coverage))


    # Save final files.
    full_errormap_lies.As[0].A = (full_errormap_lies.As[0].A*255).astype('uint8')
    h5write(opt.fwd_dir + "test_speed_" + opt.exp_name + "_" + str(opt.chkpt_num) + ".h5", img=np.reshape(full_errormap_lies.As[0].A, volume_size))
    print(str(i+1) + " samples saved!")


# Run inference
if __name__ == "__main__":

    data_dir = "/usr/people/jabae/seungmount/Omni/TracerTasks/pinky/proofreading/"

    TEST = MultiDataset(
        [
            os.path.expanduser(data_dir + "chunk_18049-20096_30337-32384_4003-4258.omni.files/")
        ],
        {
            "machine_labels": "lzf_mean_agg_tr.h5",
            "image": "image.h5"
        }
    )
    

    # Options
    opt = Options()
    
    opt.exp_dir = '/usr/people/jabae/seungmount/research/Alex/error_detection/exp/reconstruction_norm_lr_0001/exp_reconstruction_norm_loc_mip1_320_0313/'
    opt.model_dir = opt.exp_dir + 'model/'
    opt.fwd_dir = '/usr/people/jabae/seungmount/research/Alex/error_detection/test_vol/'
    opt.exp_name = 'edX_replica'

    opt.data = TEST
    opt.patch_size = (33,320,320)
    opt.out_size = (33,20,20)

    opt.net = UNet()
    opt.chkpt_num = 390000
    opt.mip = 1 
    
    opt.gpu_ids = ["0"]

    opt.in_spec = ['obj_mask','image']
    opt.out_spec = ['error2']

    opt.smp_size = 1000

    # GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu_ids)

    # Make directories.
    if not os.path.isdir(opt.fwd_dir):
        os.makedirs(opt.fwd_dir)

    # Run inference.
    print("Running inference: {}".format(opt.exp_name))

    test(opt)
