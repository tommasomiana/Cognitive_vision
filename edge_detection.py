import matplotlib.pyplot as plt
from skimage import data
import cv2
import numpy as np

#im = data.astronaut()
im = cv2.imread('data/gcp-png-3.png')
im = cv2.resize(im, (500, 500))
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', im)

# compute 2 convolutions with the 2 sobel karnel, the result is 2 tensors
#Sobelx
sobel_x = cv2.Sobel(im, cv2.CV_16S, 1, 0, ksize=3)
#sobel_x = cv2.convertScaleAbs(sobel_x)
cv2.imshow('sobel X',sobel_x)

#Sobely
sobel_y = cv2.Sobel(im, cv2.CV_16S, 0, 1, ksize=3)
#sobel_y = cv2.convertScaleAbs(sobel_y)
cv2.imshow('sobel Y',sobel_y)

# normalization of derivatives in int8
#sobel_x = sobel_x * 255 / (2**16 -1)
#sobel_y = sobel_y * 255 / (2**16 -1)

# compute the magnitude and the arctg with this two tensor
teta = np.arctan2(sobel_y, sobel_x)
# norm and shift teta
norm_teta = (teta - np.pi) * 90 / np.pi
gradient_magnitude = np.hypot(sobel_y, sobel_x)
#gradient_magnitude *= 255 / gradient_magnitude.max()
gradient_magnitude *= 255/np.sqrt(2*((255*3)**2))

# put the values in a HVG image tensor
output = np.stack((im,) * 3, axis=-1)
output[:, :, 0] = norm_teta
output[:, :, 1] = np.full(im.shape, 255)
output[:, :, 2] = gradient_magnitude

rgb_output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)
cv2.imshow('RGB', rgb_output)

cv2.waitKey(0)

