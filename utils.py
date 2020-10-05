import numpy as np
import cv2
import random

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def shuffle_set(image, label):
	per = np.random.permutation(image.shape[0])
	new_image = image[per,:,:,:]
	new_label = label[per,:,:,:]
	return new_image, new_label



def data_aug(image, label, methods_list, methods_proportion):
	assert len(methods_list)==len(methods_proportion), 'The number of selected augmentation must match the number of proportion'
	basic_number_bottom = 0
	basic_number_top = methods_proportion[0]
	seed = random.random()
	for i in range(len(methods_list)):
		if i != 0:
			basic_number_bottom = basic_number_top
			basic_number_top += methods_proportion[i]
		if seed >= basic_number_bottom and seed <= basic_number_top:
			return globals().get(methods_list[i])(image, label)
	return image, label
		
			

def random_rotate(image, label):
	(height, width, deepth) = image.shape
	seed = random.randint(-179, 179)
	M = cv2.getRotationMatrix2D((width/2,height/2),seed,1)
	image = cv2.warpAffine(image, M, (width, height))
	label = cv2.warpAffine(label, M, (width, height), flags=cv2.INTER_NEAREST)
	return image, label

def random_flip(image, label):
	seed = random.randint(-1,1)
	image = cv2.flip(image, seed)
	label = cv2.flip(label, seed)
	return image, label

def gaussian_blur(image, label):
	seed = random.uniform(0,2)
	image = cv2.GaussianBlur(image,(5,5),seed)
	return image, label

def median_blur(image, label):
	image = cv2.medianBlur(image, 3)
	return image, label

def random_unknown(image, label):
	(height, width, deepth) = image.shape
	in_seed1_1 = random.uniform(0.75,1.5)
	in_seed1_2 = random.uniform(0,0.5)
	in_seed1_3 = random.randint(-100, 100)
	in_seed1_4 = random.uniform(0,0.5)
	in_seed1_5 = random.uniform(0.75,1.5)
	in_seed1_6 = random.randint(-100, 100)
	M=np.array([[in_seed1_1,in_seed1_2,in_seed1_3],[in_seed1_4,in_seed1_5,in_seed1_6]],dtype=np.float32)
	image = cv2.warpAffine(image,M,(height,width))
	label = cv2.warpAffine(label,M,(height,width),flags=cv2.INTER_NEAREST)
	return image, image

def elastic_transform(image, label):
	total_image = np.concatenate((image, label), axis = 2)
	total_image = elastic_transform_function(total_image, total_image.shape[1] * 3, total_image.shape[1] * 0.07, total_image.shape[1] * 0.09)
	image = total_image[:,:,0:3]
	label = total_image[:,:,3:]
	return image, label


def elastic_transform_function(image, alpha, sigma, alpha_affine, random_state=None):

	if random_state is None:
		random_state = np.random.RandomState(None)

	shape = image.shape
	shape_size = shape[:2]
    
	# Random affine
	center_square = np.float32(shape_size) // 2
	square_size = min(shape_size) // 3
	pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
	pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
	M = cv2.getAffineTransform(pts1, pts2)
	image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

	dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
	dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
	dz = np.zeros_like(dx)

	x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
	indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

	return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

	





