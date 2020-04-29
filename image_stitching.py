from io import BytesIO
import numpy as np
import cv2
from skimage import data

im0 = cv2.imread("data/gallery_0.jpg")
im1 = cv2.imread("data/gallery_1.jpg")
base_img = im0
to_warp_img = im1

# Manually identify (at least) four corresponding pairs of points
"""
pixels
img0    [193,33], [318,95], [181,240], [312,212], [339,98], [334,212]
img1    [138,50], [341,58], [133,191], [335,200], [417,50], [412,212]
"""

# Estimate the homography between the first and the second image using the detected point pairs.
dst_points = np.float32([[193,33], [318,95], [181,240], [312,211], [339,98], [334,212]]) # imo
src_points = np.float32([[138,50], [341,58], [133,192], [334,200], [417,50], [412,212]]) # im1
#matrix = cv2.getPerspectiveTransform(src_points, dst_points)
matrix, status = cv2.findHomography(src_points, dst_points)

# Warp the second image using the estimated transformation matrix.
warped_img = cv2.warpPerspective(to_warp_img, matrix, (320, 600))

# "Merge" the two images in a single one by sticking one on top of the other.
totalY = warped_img.shape[0]
totalX = warped_img.shape[1] + base_img.shape[1]

result = np.zeros((totalY, totalX, 3), dtype=np.uint8)
result[:totalY, :warped_img.shape[1], :] = warped_img

base_img_cut = base_img[:, warped_img.shape[1]:, :]

mergedX = warped_img.shape[1] + base_img_cut.shape[1]
result[:base_img_cut.shape[0], warped_img.shape[1]:mergedX, :] = base_img_cut

#result = np.zeros((warped_img.shape[0], im1.shape[1] + warped_img.shape[1], 3), dtype=np.uint8)
#result[:, :warped_img.shape[1]:, :] = warped_img
#result[:im1.shape[0], :im1.shape[1], :] = im1

# cv2.imshow("vis", vis)

cv2.imshow("Image_0", im0)
#cv2.imshow("Perspective transformation", result)
cv2.imshow("Image_1", im1)
cv2.imshow("Perspective transformation", warped_img)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
