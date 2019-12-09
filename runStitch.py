import sys
sys.path.append("./depend/")
import cv2
import numpy                      as np
from matplotlib import pyplot     as plt
from ransac import ransac
from image import *

#Display matched drawings
DRAW_MATCHES = True
CUTOFF = .90
SCALE_IMAGE_SIZE = .5

# Taken from OpenCV documentation
img1_orig = cv2.imread("S1.jpg")
img2_orig = cv2.imread("S2.jpg")

img1_orig = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB)
img2_orig = cv2.cvtColor(img2_orig, cv2.COLOR_BGR2RGB)

img1_orig = cv2.resize(img1_orig, None, fx = SCALE_IMAGE_SIZE, fy = SCALE_IMAGE_SIZE) # Tried of waiting for fullsize images
img2_orig = cv2.resize(img2_orig, None, fx = SCALE_IMAGE_SIZE, fy = SCALE_IMAGE_SIZE)

img1 = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_orig, cv2.COLOR_BGR2GRAY)

if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)
akaze = cv2.AKAZE_create()

kpts1, desc1 = akaze.detectAndCompute(img1, None)
kpts2, desc2 = akaze.detectAndCompute(img2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
nn_matches = matcher.knnMatch(desc1, desc2, k=2)

matched1 = []
matched2 = []
filtered = []
for m, n in nn_matches:
    if m.distance < (1 - CUTOFF) * n.distance:
        matched1.append(kpts1[m.queryIdx].pt)
        matched2.append(kpts2[m.trainIdx].pt)
        filtered.append([m])

if (DRAW_MATCHES):
  results = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, filtered[:50], None, flags=2)

print("Number of matches: ", len(filtered))
#plt.imshow(results)
#plt.show()
# End of openCV code

#Our code:
homography = ransac(matched2, matched1, CUTOFF)

combined = combine_images(img1_orig, img2_orig, homography)

combined = trim(combined)

plt.imshow(combined)
plt.show()
