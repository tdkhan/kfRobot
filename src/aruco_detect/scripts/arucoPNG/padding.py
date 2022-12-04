#!/usr/bin/env python3

import cv2
import numpy as np
import os
import natsort
    
if __name__ == "__main__":

	path = "./png/"
	dir_list = os.listdir(path)
	images = natsort.natsorted(dir_list)
	outputPath = "./paddedAruco/"

	for image in images:

		img = cv2.imread(path+image)

		old_image_height, old_image_width, channels = img.shape
		# print(old_image_width, old_image_height)

		# create new image of desired size and color (white) for padding
		new_image_width = old_image_width + 756
		new_image_height = old_image_height + 756
		color = (255, 255, 255)
		result = np.full((new_image_height, new_image_width, channels), color, dtype = np.uint8)

		# compute center offset
		x_center = (new_image_width - old_image_width) // 2
		y_center = (new_image_height - old_image_height) // 2

		# copy old image on to the center of the new image
		result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img

		# view result
		# cv2.imshow("result", result)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		# save result
		cv2.imwrite(outputPath+image, result)

