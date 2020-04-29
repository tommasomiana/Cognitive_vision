import numpy as np

im = np.random.randint(low=0, high=255, size=(3, 7, 6))
nbin = 20

color_histogram = []
histogram = []
for c in range(im.shape[0]):
    histogram = np.zeros(nbin)
    for row in range(im.shape[1]):
        for col in range(im.shape[2]):
            pixel = im[c, row, col]
            bin = pixel * nbin // 256
            histogram[bin] += 1
    color_histogram = np.concatenate((color_histogram, histogram))

color_histogram = color_histogram / np.sum(color_histogram)
print(color_histogram)
print(np.sum(color_histogram))