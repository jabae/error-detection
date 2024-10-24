import numpy as np 
import numpy.random as random

def contrast_augment(sample):
	"""
	Performs contrast augmentation on img.

	Args:
	sample: dictionary of images ([ch,z,x,y] arrays)
	sample: dictionary of augmented images ([ch,z,x,y] arrays)
	"""

	keys = ["image"]

	# trasnformation parameters
	for k in keys:
		
		img = sample[k]
		h = img.max()
		l = img.min()

		a = random.rand()*(255/(h-l))
		b = random.randint(-a*l, 256-a*h)

		sample[k] = a*img + b


	return sample