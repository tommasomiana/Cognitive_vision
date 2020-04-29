import random
import numpy as np
from skimage import data
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)


# calculate frequency of pixels in range 0-255
#histg = cv2.calcHist(im, [0], None, [256], [0, 256])
histg,bins = np.histogram(im,256,[0,256])
bins = bins[:256]
"""
histg = np.zeros(256)
for row in range(im.shape[0]):
    for col in range(im.shape[1]):
        pixel = im[row, col]
        bin = pixel * 256 // 256
        histg[bin] += 1
"""

max = 0
for t in range(255):
    w1 = np.sum(histg[:t+1])
    w2 = np.sum(histg[t+1:])
    if w1 == 0 or w2 == 0:
        continue
    u1 = np.sum(histg[:t+1] * bins[:t+1]) / w1
    u2 = np.sum(histg[t+1:] * bins[t+1:]) / w2
    var = w1 * w2 * (u1 - u2)**2
    if var > max:
        max = var
        out = t

print(out)
