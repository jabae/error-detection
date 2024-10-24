# Import necessary packages
import os
from time import time
import argparse

import torch
import torch.nn.functional as F

from errordetector.test.model import Model
from errordetector.test.utils import *

from errordetector.utils.utils import *

from errordetector.nets.nets import *



# Pinky10 test data
data_dir = "/usr/people/jabae/seungmount/Omni/TracerTasks/pinky/proofreading/"

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


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--network", required=True,
		help="Network architecture")
	parser.add_argument("--model_dir", required=True,
		help="Model directory")
	parser.add_argument("--chkpt_num", required=True,
		help="Checkpoint number")

	opt = parser.parse_args()