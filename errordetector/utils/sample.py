#!/usr/bin/env Python3.6

import numpy as np

from errordetector.utils.utils import *
from errordetector.utils.chunk import *

from time import time

def visited_init(seg, volume_size, patch_size):

	visited = np.zeros(volume_size, dtype=np.uint8)
	
	# Mark border as visited
	visited[:patch_size[0]//2,:,:] = 1 
	visited[:,:patch_size[1]//2,:] = 1
	visited[:,:,:patch_size[2]//2] = 1
	visited[volume_size[0]-patch_size[0]//2:,:,:] = 1
	visited[:,volume_size[1]-patch_size[1]//2:,:] = 1
	visited[:,:,volume_size[2]-patch_size[2]//2:] = 1

	# Mark boundary as visited
	visited[np.where(seg==0)] = 1

	return visited


def random_coord(bbox_start, bbox_end, n=1):

	x = np.random.randint(low=bbox_start[0], high=bbox_end[0], size=n)
	y = np.random.randint(low=bbox_start[1], high=bbox_end[1], size=n)
	z = np.random.randint(low=bbox_start[2], high=bbox_end[2], size=n)

	x = np.reshape(x, (x.size,-1))
	y = np.reshape(y, (y.size,-1))
	z = np.reshape(z, (z.size,-1))

	coord = np.concatenate([x,y,z], axis=1)

	return coord


# Random sampling
def sample_objects(vol_seg, volume_size, patch_size, visited_size):

	focus_list = np.array([])
	
	visited = visited_init(vol_seg.A, volume_size, patch_size)
	vol_visited = Volume(visited, visited_size)

	print(">>>>> Sampling valid points...")
	t0 = time()

	bbox_start = [patch_size[i]//2 for i in range(3)]
	bbox_end = [volume_size[i]-patch_size[i]//2 for i in range(3)]

	cover = 0
	i = 0
	while cover < 1:

		focus = random_coord(bbox_start, bbox_end)[0]

		if vol_visited.A[focus[0],focus[1],focus[2]] >= 1:
			continue

		focus_list = np.concatenate((focus_list, focus))

		patch_seg = vol_seg[focus]
		patch_obj_mask = object_mask(patch_seg)
		patch_obj_mask_crop = patch_obj_mask[tuple([
			slice(patch_size[i]//2-visited_size[i]//2, patch_size[i]//2+visited_size[i]//2)
			for i in range(len(patch_size))])]

		vol_visited[focus] = vol_visited[focus] + patch_obj_mask_crop

		cover = np.round(np.sum(vol_visited.A>=1)/np.prod(volume_size),4)

		i += 1

		if i % 100 == 0 or i <= 10:
			print("{} covered.".format(cover))		

	focus_list = np.reshape(focus_list,(-1,3)).astype(np.uint32)

	elapsed = np.round(time()-t0, 3)
	print(">>>>> Sampling complete!")
	print("Elapsed time = {}".format(elapsed))
	
	return focus_list


# def sample_objects_chunked(vol_seg, volume_size, patch_size, visited_size, chunk_size):

# 	visited = visited_init(vol_seg.A, volume_size, patch_size)
# 	vol_visited = Volume(visited, visited_size)

# 	print(">>>>> Sampling valid points...")
# 	t0 = time()

# 	focus_list = np.array([])
	
# 	bbox_start = [patch_size[i]//2 for i in range(3)]
# 	bbox_end = [volume_size[i]-patch_size[i]//2 for i in range(3)]

# 	bbox_chunks = chunk_bboxes(bbox_start, bbox_end, chunk_size)
	
# 	for bbox in bbox_chunks:

# 		bbox_start_chunk = bbox[0]
# 		bbox_end_chunk = bbox[1]
		
# 		cover = 0
# 		i = 0
# 		while cover < 1:

# 			focus = random_coord(bbox_start_chunk, bbox_end_chunk)[0]

# 			if vol_visited.A[focus[0],focus[1],focus[2]] >= 1:
# 				continue

# 			focus_list = np.concatenate((focus_list, focus))

# 			patch_seg = vol_seg[focus]
# 			patch_obj_mask = object_mask(patch_seg)
# 			patch_obj_mask_crop = patch_obj_mask[tuple([
# 				slice(patch_size[i]//2-visited_size[i]//2, patch_size[i]//2+visited_size[i]//2)
# 				for i in range(len(patch_size))])]

# 			vol_visited[focus] = vol_visited[focus] + patch_obj_mask_crop

# 			n_covered = np.sum(vol_visited.A[tuple([slice(bbox_start_chunk[i], bbox_end_chunk[i])
# 																		for i in range(3)])]>=1)
# 			chunk_size = np.prod([bbox_end_chunk[i]-bbox_start_chunk[i] for i in range(3)])
# 			cover = np.round(n_covered/chunk_size,4)

# 			i += 1

# 			if i % 100 == 0 or i <= 10:
# 				print("{} covered.".format(cover))

# 	focus_list = np.reshape(focus_list,(-1,3)).astype(np.uint32)

# 	elapsed = np.round(time()-t0, 3)
# 	print(">>>>> Sampling complete!")
# 	print("Elapsed time = {}".format(elapsed))
	
# 	return focus_list 


from dataset import *

def sample_objects_chunked(vol_seg, volume_size, patch_size, visited_size, chunk_size, mip=0):

	seg = vol_seg.A

	if mip > 0:
		mip_factor = 2**mip
		
		seg = seg[:,::mip_factor,::mip_factor]
		volume_size = (volume_size[0],
										volume_size[1]//mip_factor,
										volume_size[2]//mip_factor)
		patch_size = (patch_size[0],
										patch_size[1]//mip_factor,
										patch_size[2]//mip_factor)
		visited_size = (visited_size[0],
										visited_size[1]//mip_factor,
										visited_size[2]//mip_factor)
		chunk_size = (chunk_size[0],
										chunk_size[1]//mip_factor,
										chunk_size[2]//mip_factor)

	visited = visited_init(seg, volume_size, patch_size)
	vol_visited = Volume(visited, visited_size)
	vol_seg = Volume(seg, patch_size)
	
	print(">>>>> Sampling valid points...")
	t0 = time()

	focus_list = np.array([])
	
	bbox_start = [patch_size[i]//2 for i in range(3)]
	bbox_end = [volume_size[i]-patch_size[i]//2 for i in range(3)]

	bbox_chunks = chunk_bboxes(bbox_start, bbox_end, chunk_size)
	
	for bbox in bbox_chunks:

		bbox_start_chunk = bbox[0]
		bbox_end_chunk = bbox[1]
		
		cover = 0
		i = 0
		while cover < 1:

			focus = random_coord(bbox_start_chunk, bbox_end_chunk)[0]

			if vol_visited.A[focus[0],focus[1],focus[2]] >= 1:
				continue

			focus_list = np.concatenate((focus_list, focus))

			patch_seg = vol_seg[focus]
			patch_obj_mask = object_mask(patch_seg)
			patch_obj_mask_crop = patch_obj_mask[tuple([
				slice(patch_size[i]//2-visited_size[i]//2, patch_size[i]//2+visited_size[i]//2)
				for i in range(len(patch_size))])]

			vol_visited[focus] = vol_visited[focus] + patch_obj_mask_crop

			n_covered = np.sum(vol_visited.A[tuple([slice(bbox_start_chunk[i], bbox_end_chunk[i])
																		for i in range(3)])]>=1)
			chunk_size = np.prod([bbox_end_chunk[i]-bbox_start_chunk[i] for i in range(3)])
			cover = np.round(n_covered/chunk_size, 3)
			
			i += 1

			if i % 100 == 0 or i <= 10:
				print("{} covered.".format(cover))

	# h5write("visited.h5",vol_visited.A.astype('uint32'))
	focus_list = np.reshape(focus_list,(-1,3)).astype(np.uint32)
	focus_list[:,1] = focus_list[:,1]*mip_factor
	focus_list[:,2] = focus_list[:,2]*mip_factor

	elapsed = np.round(time()-t0, 3)
	print(">>>>> Sampling complete!")
	print("Elapsed time = {}".format(elapsed))
	
	return focus_list 