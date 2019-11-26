import cv2
import numpy                      as np
from matplotlib import pyplot     as plt
from ransac import ransac

# Taken from OpenCV documentation
img1 = cv2.imread(cv2.samples.findFile("S1.jpg"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(cv2.samples.findFile("S2.jpg"), cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)
akaze = cv2.AKAZE_create()
kpts1, desc1 = akaze.detectAndCompute(img1, None)
kpts2, desc2 = akaze.detectAndCompute(img2, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
nn_matches = matcher.match(desc1, desc2)
nn_matches = sorted(nn_matches, key = lambda x:x.distance)
matched1 = []
matched2 = []
for m in nn_matches:
    matched1.append(kpts1[m.queryIdx].pt)
    matched2.append(kpts2[m.trainIdx].pt)



# End of openCV code

#Our code:
homography = ransac(matched1, matched2)
