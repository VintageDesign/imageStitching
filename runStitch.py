import cv2
import numpy                      as np
from matplotlib import pyplot     as plt
from checkColor import checkColor
from match      import match

# Taken from OpenCV documentation
img1 = cv2.imread(cv2.samples.findFile("S1.jpg"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(cv2.samples.findFile("S2.jpg"), cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)
akaze = cv2.AKAZE_create()
kpts1, desc1 = akaze.detectAndCompute(img1, None)
kpts2, desc2 = akaze.detectAndCompute(img2, None)
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
nn_matches = matcher.knnMatch(desc1, desc2, 2)
matched1 = []
matched2 = []
nn_match_ratio = 0.8 # Nearest neighbor matching ratio
for m, n in nn_matches:
    if m.distance < nn_match_ratio * n.distance:
        matched1.append(kpts1[m.queryIdx])
        matched2.append(kpts2[m.trainIdx])
matched1 = cv2.KeyPoint_convert(matched1)
matched2 = cv2.KeyPoint_convert(matched2)

# End of openCV code



#Our code:
homography = ransac(matched1, matched2)
