import torch

from dataset import *

from utils.utils import *



data_dir = "/usr/people/jabae/seungmount/Omni/TracerTasks/pinky/proofreading/"


TEST = MultiDataset(
        [
            os.path.expanduser(data_dir + "chunk_18049-20096_30337-32384_4003-4258.omni.files/")
        ],
        {
            "machine_labels": "lzf_mean_agg_tr.h5",
            "human_labels": "lzf_proofread.h5",
            "image": "image.h5",
            "samples": "padded_valid_samples.h5"
        }
)


padded_patch_size = (1,25,256,256)
full_labels_lies = MultiVolume(TEST.machine_labels, padded_patch_size)
full_labels_truth = MultiVolume(TEST.human_labels, padded_patch_size)
samples = MultiVolume(TEST.samples, (1,3), indexing='CORNER')


focus_list = np.array([])
for i in range(TEST.samples[0].shape[0]):

	focus_raw = samples[0, (i,0)]
	focus = torch.cat((torch.zeros((1,), dtype=focus_raw.dtype), torch.reshape(focus_raw, (3,))), 0)

	sample_human_label = full_labels_truth[0, focus]
	
	sample_machine_label = full_labels_lies[0, focus]
	sample_obj_mask = object_mask(sample_machine_label)

	error_inner = has_error(sample_obj_mask[:,10:14,108:148,108:148], sample_human_label[:,10:14,108:148,108:148],(1,1,1,1))
	error_outer = has_error(sample_obj_mask[:,8:16,88:168,88:168], sample_human_label[:,8:16,88:168,88:168],(1,1,1,1))

	if error_inner == 0 and error_outer == 1:
		continue

	focus_list = np.concatenate((focus_list, np.reshape(focus_raw.numpy(),(-1,))))


focus_array = np.reshape(focus_list, (-1,3))
h5write(data_dir + "chunk_18049-20096_30337-32384_4003-4258.omni.files/" + "padded_valid_filtered_samples.h5", img=focus_array)