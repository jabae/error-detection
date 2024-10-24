# Import necessary packages
import os
from time import time
import argparse

import torch

from errordetector.train.model import Model
from errordetector.train.logger import Logger
from errordetector.train.utils import *

from errordetector.utils.utils import *

from errordetector.nets.nets import *


def train(opt):   

	# Load model
	net = opt.net
	model = load_model(opt)

	optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
	print(optimizer)

	# Initial checkpoint
	save_chkpt(model, opt.model_dir, opt.chkpt_num)

	# Training loop
	print("========== BEGIN TRAINING LOOP ==========")
	with Logger(opt) as logger:
		# Log parameters
		logger.log_parameters(vars(opt))
		logger.log_command()

		chkpt_epoch = opt.chkpt_num//int(opt.n_train/opt.batch_size)
		chkpt_iter = opt.chkpt_num%int(opt.n_train/opt.batch_size)

		i = opt.chkpt_num
		for epoch in range(chkpt_epoch, opt.max_epoch):

			# Data loaders (Reset every epoch)
			train_loader = load_data(opt.data, opt.train_vol, opt.augment, opt)
			val_loader = load_data(opt.data, opt.test_vol, [], opt)

			for it in range(chkpt_iter, int(opt.n_train/opt.batch_size)):
				# Timer
				t0 = time()

				# Load Training samples.
				sample = train_loader()

				# Optimizer step
				optimizer.zero_grad()
				losses, preds = forward(model, sample, opt.batch_size)
				total_loss = sum(losses[k] for k in opt.out_spec)
				losses["all"] = total_loss/len(opt.out_spec)
				total_loss.backward()
				optimizer.step()

				# Elapsed time
				elapsed = time() - t0
        
				# Record keeping
				logger.record('train', losses, elapsed=elapsed)

				# Log & display averaged stats
				if (i+1) % opt.avgs_intv == 0 or i < opt.warm_up:
					logger.check('train', i+1)

				# Logging images
				if (i+1) % opt.imgs_intv == 0:
					logger.log_images('train', i+1, preds, sample)

				# Evaluation loop
				if (i+1) % opt.eval_intv == 0:
					eval_loop(i+1, model, val_loader, opt, logger)

				# Model checkpoint
				if (i+1) % opt.chkpt_intv == 0:
					save_chkpt(model, opt.model_dir, i+1)

				# Reset timer.
				t0 = time()

				i = i + 1


def eval_loop(iter_num, model, data_loader, opt, logger):

	# Evaluation loop
	print("---------- BEGIN EVALUATION LOOP ----------")
	with torch.no_grad():
		t0 = time()
		for i in range(opt.eval_iter):
			sample = data_loader()
			losses, preds = forward(model, sample, opt.batch_size)
			losses["all"] = sum(losses[k] for k in opt.out_spec)/len(opt.out_spec)
			elapsed = time() - t0

			# Record keeping
			logger.record('test', losses, elapsed=elapsed)

			# Restart timer.
			t0 = time()

		# Log & display averaged stats.
		logger.check('test', iter_num)
		print("-------------------------------------------")

		model.eval()


# Pinky10 training data
data_dir = "/usr/people/jabae/seungmount/Omni/TracerTasks/pinky/proofreading/"

TRAIN = MultiDataset(
  [
    os.path.expanduser(data_dir + "chunk_14977-17024_27265-29312_4003-4258.omni.files/"),
    os.path.expanduser(data_dir + "chunk_14977-17024_28801-30848_4003-4258.omni.files/"),
    os.path.expanduser(data_dir + "chunk_16513-18560_27265-29312_4003-4258.omni.files/"),
    os.path.expanduser(data_dir + "chunk_16513-18560_28801-30848_4003-4258.omni.files/"),
    os.path.expanduser(data_dir + "chunk_14977-17024_30337-32384_4003-4258.omni.files/"),
    os.path.expanduser(data_dir + "chunk_18049-20096_27265-29312_4003-4258.omni.files/"),
    os.path.expanduser(data_dir + "chunk_16513-18560_30337-32384_4003-4258.omni.files/")
  ],
  {
    "machine_labels": "lzf_mean_agg_tr.h5",
    "human_labels": "lzf_proofread.h5",
    "image": "image.h5",
    "samples": "padded_valid_samples.h5"
  }
)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--exp_dir", required=True,
		help="Experiment directory")
	parser.add_argument("--network", required=True,
		help="Network architecture")
	parser.add_argument("--patch_size", nargs="+", type=int, required=True,
		help="Input patch size")
	parser.add_argument("--out_size", nargs="+", type=int, required=True,
		help="Output size")
	parser.add_argument("--chkpt_num", nargs="?", type=int, required=True,
		help="Checkpoint number to start training")
	parser.add_argument("--gpu_ids", nargs="+", required=True,
		help="GPU ids for train")
	parser.add_argument("--augment", nargs="+", required=True,
		help="Data augmentation")
	parser.add_argument("--pretrain", required=False, default=None,
		help="Pretrained model directory")
	parser.add_argument("--lr", nargs="?", type=float, required=False, default=0.0001,
		help="Learning rate")

	opt = parser.parse_args()

	
	# Experiment directory
	opt.log_dir = opt.exp_dir + "/log/"
	opt.model_dir = opt.exp_dir + "/model/"

	# Training data
	opt.data = TRAIN
	opt.train_vol = [0,1,2,3,4,5]
	opt.test_vol = [6]
	opt.n_train = 1500000

	# Network architecture
	if opt.network == "UNetMip0":
		opt.net = UNetMip0()
		opt.mip = 0
	elif opt.network == "UNetMip1":
		opt.net = UNetMip1()
		opt.mip = 1
	elif opt.network == "UNetMip2":
		opt.net = UNetMip2()
		opt.mip = 2
	elif opt.network == "UNetMip3":
		opt.net = UNetMip3()
		opt.mip = 3
	elif opt.network == "MergeNet":
		opt.net = MergeNet()
		opt.mip = 4
	else:
		raise Exception("Network architecture doesn't exist.")

	# Train parameters
	opt.in_spec = sorted(["occluded", "obj_mask", "image"])
	opt.out_spec = sorted(["reconstruct", "error"])
	
	opt.max_epoch = 5
	opt.max_iter = 1000000

	# Log parameters
	opt.chkpt_intv = 3000
	opt.avgs_intv = 100
	opt.imgs_intv = 1000
	opt.warm_up = 100

	opt.eval_iter = 100
	opt.eval_intv = 1000

	# Set GPU environment
	opt.batch_size = len(opt.gpu_ids)
	opt.num_workers = len(opt.gpu_ids)
	os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(opt.gpu_ids)

	# Setup experiment directory
	if not os.path.isdir(opt.exp_dir):
		os.makedirs(opt.exp_dir)
	if not os.path.isdir(opt.log_dir):
		os.makedirs(opt.log_dir)
	if not os.path.isdir(opt.model_dir):
		os.makedirs(opt.model_dir)

	# Run train
	print("Running experiment: {}".format(opt.exp_dir))
	train(opt)
