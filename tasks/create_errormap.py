#!/usr/bin/env python3
"""
Create errormap wrapper script
"""

# Import necessary packages
import argparse

import errordetector as errdet


parser = argparse.ArgumentParser()

# IO
parser.add_argument("out_cvname")

# Parameters
parser.add_argument("--mip", nargs="?", type=int, required=True)
parser.add_argument("--vol_shape", nargs="+", type=int, required=True)
parser.add_argument("--chunk_shape", nargs="+", type=int, required=True)
parser.add_argument("--patch_shape", nargs="+", type=int, required=True)
parser.add_argument("--offset", nargs="+", type=int, required=True)

opt = parser.parse_args()

# Run
errdet.tasks.create_errormap_task(opt)