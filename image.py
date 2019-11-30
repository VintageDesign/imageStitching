"""
This module is used to manage the image editing
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

def trim(image):
    if not np.sum(image[0]):
        return trim(image[1:])
    if not np.sum(image[-1]):
        return trim(image[:-2])
    if not np.sum(image[:,0]):
        return trim(image[:,1:])
    if not np.sum(image[:,-1]):
        return trim(image[:,:-2])
    return image

def combine_images(img1, img2, h_matrix):
    # Calculate new size
    try:
        (rows, columns, channels) = img1.shape
    except:
        (rows, columns) = img1.shape
        channels = 1
    print(str(columns) + "x" + str(rows) + "x" + str(channels))

    pts = np.float32([[0, 0], [0, rows-1], [columns-1, rows-1], [columns-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, h_matrix)

    dst = cv2.warpPerspective(img2, h_matrix, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    dst[0:img1.shape[0], 0:img1.shape[1]] = img1

    return dst
