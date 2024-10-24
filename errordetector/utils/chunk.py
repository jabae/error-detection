import itertools
import operator

def bounds1D(start, end, step_size, overlap=0):
  
  assert step_size > 0, "Invalid step_size: {}".format(step_size)
  assert end > start, "Invalid range: {} ~ {}".format(start, end)
  assert overlap >=0, "Invalid overlap: {}".format(overlap)

  s = start
  e = s + step_size

  bounds = []
  while e < end:
    
    bounds.append((s, e))

    s += step_size - overlap
    e = s + step_size

  e = end
  bounds.append((s,e))

  return bounds


def chunk_bboxes(bbox_start, bbox_end, chunk_size, overlap=(0,0,0)):

	x_bnds = bounds1D(bbox_start[0], bbox_end[0], chunk_size[0], overlap[0])
	y_bnds = bounds1D(bbox_start[1], bbox_end[1], chunk_size[1], overlap[1])
	z_bnds = bounds1D(bbox_start[2], bbox_end[2], chunk_size[2], overlap[2])

	bboxes = [tuple(zip(xs, ys, zs))
						for (xs, ys, zs) in itertools.product(x_bnds, y_bnds, z_bnds)]

	return bboxes