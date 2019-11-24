def match(image1, image2):
    '''
    This function performs harris feature detection on each image, then matches the points

    Input:
    npArray image1
    npArray image2

    Output:
    ?????? matchedFeatures
    '''

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)






