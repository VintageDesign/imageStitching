import cv2


# Read in as <x, y, color>
im1 = cv2.imread("S1.jpg")
im2 = cv2.imread("S2.jpg")


im1, im2 = checkColor(im1, im2)

matches = match(im1, im2)


