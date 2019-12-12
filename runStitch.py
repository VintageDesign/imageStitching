import sys
sys.path.append("./depend/")
import cv2
import numpy                      as np
from matplotlib import pyplot     as plt
import ransac
import image
from timeit import default_timer as timer

#Constants for options
DRAW_MATCHES = False #Draw mached points image
DRAW_COMBINED = True #Draw the final combined image
DISPLAY_INDIVIDUAL_TIMING = True #Draw timings for each part
CUTOFF = .95 # Cutoff for ransac/point matching
SCALE_IMAGE = True #Scale the image down in size
SCALE_IMAGE_SIZE = .5 #percent to scale the image by
MAX_RANSAC = 10000 #Max number of ransac iterations

# If we're not scaling pretend scale value is 1
if (SCALE_IMAGE is False): 
  SCALE_IMAGE_SIZE = 1

#Check to see if we were given photos to stitch
if (len(sys.argv) == 3):
  file1 = sys.argv[1]
  file2 = sys.argv[2]
elif (len(sys.argv) != 1):
  print("Invalid number of system arguments defaulting to S1.jpg and S2.jpg")
  file1 = "S1.jpg"
  file2 = "S2.jpg"
else:
  file1 = "S1.jpg"
  file2 = "S2.jpg"

# Taken from OpenCV documentation
totalstart = timer()
img1_orig = cv2.imread(file1)  # Read image file 1
img2_orig = cv2.imread(file2)  # Read image file 2

img1_orig = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB) # conver the image to RGB because opencv is weird
img2_orig = cv2.cvtColor(img2_orig, cv2.COLOR_BGR2RGB) # and loads them as BGR

# Tired of waiting for fullsize images so scale them
if (SCALE_IMAGE is False): 
  img1_orig = cv2.resize(img1_orig, None, fx = SCALE_IMAGE_SIZE, fy = SCALE_IMAGE_SIZE)
  img2_orig = cv2.resize(img2_orig, None, fx = SCALE_IMAGE_SIZE, fy = SCALE_IMAGE_SIZE)

# Create a grayscale copy of the image to speed up the processing time
img1 = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_orig, cv2.COLOR_BGR2GRAY)
totalend = timer()
if (DISPLAY_INDIVIDUAL_TIMING):
  print("Image import runtime: " + str(totalend-totalstart))

if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)
akaze = cv2.AKAZE_create()                       # Create a AKAZE feature detector object

start = timer()
kpts1, desc1 = akaze.detectAndCompute(img1, None) # Detect features in image 1
end = timer()
if (DISPLAY_INDIVIDUAL_TIMING):
  print("Feature detect(img1) runtime: " + str(end-start))
start = timer()
kpts2, desc2 = akaze.detectAndCompute(img2, None) # Detect features in image 2
end = timer()
if (DISPLAY_INDIVIDUAL_TIMING):
  print("Feature detect(img2) runtime: " + str(end-start))

start = timer()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)         #Create a matcher object
nn_matches = matcher.knnMatch(desc1, desc2, k=2)  # Perform k-nearest neighbor matching 
end = timer()
if (DISPLAY_INDIVIDUAL_TIMING):
  print("Feature match runtime: " + str(end-start))

start = timer()
matched1 = []
matched2 = []
filtered = []
# Do our own ratio test removal of points
for m, n in nn_matches:
    if m.distance < (1 - CUTOFF) * n.distance: # Only save matches that are above our cutoff (non-ambiguous points)
        matched1.append(kpts1[m.queryIdx].pt)
        matched2.append(kpts2[m.trainIdx].pt)
        filtered.append([m])  # this is used for when we draw matches 
end = timer()
if (DISPLAY_INDIVIDUAL_TIMING):
  print("Ratio Test runtime: " + str(end-start))

if (DRAW_MATCHES):
  results = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, filtered[:50], None, flags=2) # Create image
  plt.imshow(results)                                                                  # with our matches
  plt.show()                                                                           # linked together

print("\tNumber of matches: ", len(filtered))
# End of openCV code

#Our code:
start = timer()
homography = ransac.ransac(matched2, matched1, CUTOFF, MAX_RANSAC) # Perform the RANSAC
end = timer()
if (DISPLAY_INDIVIDUAL_TIMING):
  print("RANSAC runtime: " + str(end-start))

start = timer()
combined = image.combine_images(img1_orig, img2_orig, homography) # Stitch the two images together
combined = image.trim(combined)                                   # Trim off blank space around the image
end = timer()                                                     # we used as padding
if (DISPLAY_INDIVIDUAL_TIMING):
  print("Create new image runtime: " + str(end-start))

totalend = timer()
print("Total runtime: " + str(totalend - totalstart))

if (DRAW_COMBINED):
  plt.imshow(combined) # Show the image on screen
  plt.show()

cv2.imwrite('combined.jpg', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)) # write the image to a file 
                                                                       # (and convert back to weird BGR format)

