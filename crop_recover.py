import numpy as np
def calculate_crop_number(image, crop_height, crop_width, oc):
	height = image.shape[0]
	width = image.shape[1]
	height_number = height // crop_height + 1
	height_number = oc * (height_number - 1) + 1
	width_number = width // crop_width + 1
	width_number = oc * (width_number - 1) + 1
	output = height_number * width_number
	return output, height_number, width_number


def test_and_complement(image, crop_height, crop_width):
	if image.shape[0] != crop_height or image.shape[1] != crop_width:
		complement = np.zeros([crop_height, crop_width, image.shape[2]]).astype(np.float32)
		complement[0:image.shape[0], 0:image.shape[1], :] = image
		return complement
	else:
		return image

def crop_image(image, crop_height = 256, crop_width = 256, oc = 2):
	total_output_number, height_number, width_number = calculate_crop_number(image, crop_height, crop_width, oc)
	output = np.zeros([total_output_number, crop_height, crop_width, image.shape[2]]).astype(np.float32)
	count = 0
	for i in range(height_number):
		for j in range(width_number):
			unit_crop_image = image[int(crop_height/oc*i):int(crop_height/oc*i)+crop_height,
						int(crop_width/oc*j):int(crop_width/oc*j)+crop_width,:]
			unit_crop_image = test_and_complement(unit_crop_image, crop_height, crop_width)
			output[count] = unit_crop_image
			count += 1
	return output
			




def recover_image(cropped_image, height, width, crop_height = 256, crop_width = 256, oc = 2):
	
	in_height_number = height // crop_height + 1
	height_number = oc * (in_height_number - 1) + 1
	in_width_number = width // crop_width + 1
	width_number = oc * (in_width_number - 1) + 1
	output_image = np.zeros([in_height_number*crop_height, in_width_number*crop_width, cropped_image.shape[3]]).astype(np.float32)
	assert crop_height * (oc - 1) % (2 * oc) == 0 and crop_width * (oc - 1) % (2 * oc) == 0,\
	'The input crop image size and overlap coefficient cannot meet the exact division'
	h_sec_pos = int(crop_height * (oc - 1) / (2 * oc))
	w_sec_pos = int(crop_width * (oc - 1) / (2 * oc))
	h_thi_pos = int(crop_height * (oc + 1) / (2 * oc))
	w_thi_pos = int(crop_width * (oc + 1) / (2 * oc))
	h_half_pos = int(crop_height/oc)
	w_half_pos = int(crop_width/oc)

	for i in range(height_number):
		if i == 0:
			for j in range(width_number):
				if height_number == 1:
					if j == 0:
						if width_number == 1:
							output_image[0:crop_height,0:crop_width,:]=\
							cropped_image[i*width_number+j][0:crop_height,0:crop_width,:]
						else:
							output_image[0:crop_height,0:w_thi_pos,:]=\
							cropped_image[i*width_number+j][0:crop_height,0:w_thi_pos,:]
					elif j == (width_number -1):
						output_image[0:crop_height,j*w_half_pos+w_sec_pos:,:] =\
						 cropped_image[i*width_number+j][0:crop_height,w_sec_pos:crop_width,:]
					else:
						output_image[0:crop_height,w_thi_pos+(j-1)*w_half_pos:w_thi_pos+j*w_half_pos,:] =\
						cropped_image[i*width_number+j][0:crop_height,w_sec_pos:w_thi_pos,:]

				else:
					if j == 0:
						if width_number == 1:
							output_image[0:h_thi_pos,0:crop_width,:]=\
							cropped_image[i*width_number+j][0:h_thi_pos,0:crop_width,:]
						else:
							output_image[0:h_thi_pos,0:w_thi_pos,:]=\
							cropped_image[i*width_number+j][0:h_thi_pos,0:w_thi_pos,:]
					elif j == (width_number -1):
						output_image[0:h_thi_pos,j*w_half_pos+w_sec_pos:,:] =\
						cropped_image[i*width_number+j][0:h_thi_pos,w_sec_pos:crop_width,:]
					else:
						output_image[0:h_thi_pos,w_thi_pos+(j-1)*w_half_pos:w_thi_pos+j*w_half_pos,:] =\
						cropped_image[i*width_number+j][0:h_thi_pos,w_sec_pos:w_thi_pos,:]
		elif i == (height_number - 1):
			for j in range(width_number):
				if j == 0:
					if width_number == 1:
						output_image[i*h_half_pos+h_sec_pos:,0:crop_width,:]=\
						cropped_image[i*width_number+j][h_sec_pos:crop_height,0:crop_width,:]
					else:
						output_image[i*h_half_pos+h_sec_pos:,0:w_thi_pos,:]=\
						cropped_image[i*width_number+j][h_sec_pos:crop_height,0:w_thi_pos,:]
				elif j == (width_number - 1):
					output_image[i*h_half_pos+h_sec_pos:,j*w_half_pos+w_sec_pos:,:] =\
					cropped_image[i*width_number+j][h_sec_pos:crop_height,w_sec_pos:crop_width,:]
				else:
					output_image[i*h_half_pos+h_sec_pos:,w_thi_pos+(j-1)*w_half_pos:w_thi_pos+j*w_half_pos,:] =\
					cropped_image[i*width_number+j][h_sec_pos:crop_height,w_sec_pos:w_thi_pos,:]
		else:
			for j in range(width_number):
				if j == 0:
					if width_number == 1:
						output_image[h_thi_pos+(i-1)*h_half_pos:h_thi_pos+i*h_half_pos,
						0:crop_width,:]=cropped_image[i*width_number+j][h_sec_pos:h_thi_pos,0:crop_width,:]
					else:
						output_image[h_thi_pos+(i-1)*h_half_pos:h_thi_pos+i*h_half_pos,
						0:w_thi_pos,:]=cropped_image[i*width_number+j][h_sec_pos:h_thi_pos,0:w_thi_pos,:]
				elif j == (width_number - 1):
					output_image[h_thi_pos+(i-1)*h_half_pos:h_thi_pos+i*h_half_pos,j*w_half_pos+w_sec_pos:,:] =\
					cropped_image[i*width_number+j][h_sec_pos:h_thi_pos,w_sec_pos:crop_width,:]
				else:
					output_image[h_thi_pos+(i-1)*h_half_pos:h_thi_pos+i*h_half_pos,
					w_thi_pos+(j-1)*w_half_pos:w_thi_pos+j*w_half_pos,:] = \
					cropped_image[i*width_number+j][h_sec_pos:h_thi_pos,w_sec_pos:w_thi_pos,:]
	output_image = output_image[0:height,0:width,:]
	return output_image
			


	




	
	
	
	
			
			
	

	





