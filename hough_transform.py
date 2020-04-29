import numpy as np
import cv2
from skimage import data

im = data.coins()[160:230, 70:270]
cv2.imshow("im", im)

edges = cv2.Canny(im, 320, 500)
cim = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
cv2.imshow("edges", edges)

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=20, minRadius=0, maxRadius=100)

circles = np.uint16(np.around(circles))
if circles is not None:
    for (x, y, r) in circles[0, :]:
        # draw the outer circle
        cv2.circle(cim, (x, y), r, (0,255,0), 2)
        # draw the center of the circle
        cv2.circle(cim, (x, y), 1, (0,0,255), 3)

#cv2.imshow('detected circles', edges)
cv2.imshow("output", cim)
cv2.waitKey()