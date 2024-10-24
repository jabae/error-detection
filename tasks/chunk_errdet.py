#!/usr/bin/env python3
"""
Chunkwise error detector wrapper script
"""

# Import necessary packages
import argparse

import errordetector as errdet


parser = argparse.ArgumentParser()

# IO
parser.add_argument("seg_cvname")
parser.add_argument("img_cvname")
parser.add_argument("out_cvname")

# Parameters 
parser.add_argument("--chunk_begin_seg", nargs="+", type=int, required=True)
parser.add_argument("--chunk_end_seg", nargs="+", type=int, required=True)
parser.add_argument("--chunk_begin_img", nargs="+", type=int, required=True)
parser.add_argument("--chunk_end_img", nargs="+", type=int, required=True)
parser.add_argument("--mip", nargs="?", type=int, required=True)
parser.add_argument("--model_dir", required=True)
parser.add_argument("--chkpt_num", nargs="?", type=int, required=True)
parser.add_argument("--patch_shape", nargs="+", type=int, required=True)
parser.add_argument("--out_shape", nargs="+", type=int, required=True)

opt = parser.parse_args()

# Run
errdet.tasks.chunk_errdet_task(opt)