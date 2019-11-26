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

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
nn_matches = matcher.knnMatch(desc1, desc2, k=2)

matched1 = []
matched2 = []
filtered = []
for m, n in nn_matches:
    if m.distance < 0.10 * n.distance:
        matched1.append(kpts1[m.queryIdx].pt)
        matched2.append(kpts2[m.trainIdx].pt)
        filtered.append([m])

results = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, filtered[:50], None, flags=2)

print("Number of matches: ", len(filtered))
#plt.imshow(results)
#plt.show()
# End of openCV code

#Our code:
homography = ransac(matched1, matched2)
