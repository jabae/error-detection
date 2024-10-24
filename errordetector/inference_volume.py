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

from nets.unet_all import *

from utils.utils import *


def test_samples(opt):

    # Load model
    model = load_model(opt)
    
    # Load test data
    machine_labels = opt.data.machine_labels
    image = opt.data.image
    samples = opt.data.samples
    
    volume_size = human_labels[0].shape[1:]
    padded_volume_size = human_labels[0].shape
    patch_size = opt.patch_size
    padded_patch_size = (1,) + patch_size

    error_map = [np.zeros(padded_volume_size)]
    
    full_image = MultiVolume(image, padded_patch_size)
    full_labels_lies = MultiVolume(machine_labels, padded_patch_size)
    full_error_map_lies = MultiVolume(error_map, padded_patch_size)
    full_samples = MultiVolume(samples, (1,3), indexing='CORNER')

    
    for i in range(samples[0].shape[0]):

        t0 = time.time()

        focus = full_samples[0, (i,0)]
        focus = torch.cat((torch.zeros((1,), dtype=focus.dtype), torch.reshape(focus, (3,))), 0)
        
        # Prepare sample
        sample_machine_label = full_labels_lies[0, focus]
        obj_id = sample_machine_label[0,int(padded_patch_size[1]/2),int(padded_patch_size[2]/2),int(padded_patch_size[3]/2)]
        sample_obj_mask = object_mask(sample_machine_label)
        sample_image = full_image[0, focus]
  
        
        sample = {}
        sample["obj_mask"] = torch.reshape(sample_obj_mask, (1,)+sample_obj_mask.shape)
        sample["image"] = torch.reshape(sample_image, (1,)+sample_image.shape)

        # Discriminate
        preds = forward(model, sample)

        # Predictions
        pred2 = preds["error2"]

        pred2_upsample = F.upsample(torch.reshape(pred2, (1,25,16,16)), scale_factor=(16,16), mode='bilinear').cpu().detach().numpy()
        full_error_map_lies[0, focus] = np.maximum(full_error_map_lies[0, focus], pred2_upsample*sample["obj_mask"])

        # Stats
        elapsed = np.round(time.time() - t0, 3)


        # Print iteration stat
        if (i+1) % 100 == 0 or (i+1) <= 10:
            print("Iter:  " + str(i+1) + "/" + str(samples[0].shape[0]) + ", elapsed time = " + str(elapsed))


    # Save final files.
    full_error_map_lies.As[0].A = (full_error_map_lies.As[0].A*255).astype('uint8')

    h5write(opt.fwd_dir + "pred_errormap_basil_" + str(opt.chkpt_num) + ".h5", img=np.reshape(full_error_map_lies.As[0].A, volume_size))
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
            "machine_labels": "basil_seg.h5",
            "human_labels": "lzf_proofread.h5",
            "image": "image.h5",
        }
    )


    samples = h5read('/usr/people/jabae/seungmount/research/Alex/errordetection/test_vol/test_samples_33_320_320.h5')
    TEST.samples = [samples]

    

    opt = Options()
    
    opt.exp_dir = '/usr/people/jabae/seungmount/research/Alex/errordetection/exp/reconstruction_norm_lr_0001/exp_reconstruction_norm_contrast_augment_error2_320_1024/'
    opt.model_dir = opt.exp_dir + 'model/'
    opt.fwd_dir = '/usr/people/jabae/seungmount/research/Alex/errordetection/test_vol/'
    opt.exp_name = 'Inference'

    opt.data = TEST
    opt.patch_size = (33,320,320)
    opt.out_size = (33,20,20)

    opt.net = UNet()
    opt.chkpt_num = 219000
    
    opt.gpu_ids = ["0"]

    opt.in_spec = ['obj_mask','image']
    opt.out_spec = ['error2']

    opt.chkpt_intv = 3000

	# GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu_ids)

    # Make directories.
    if not os.path.isdir(opt.fwd_dir):
        os.makedirs(opt.fwd_dir)

    # Run inference.
    print("Running inference: {}".format(opt.exp_name))

    test_samples(opt)

