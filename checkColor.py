import cv2

def checkColor(image1, image2):
    '''
    This function will make sure that both images are rgb if they are 2 different formats, the
    images are converted to grey scale.


    inputs:
    npArray image1
    npArray image2

    returns:
    npArray retImage1
    npArray retImage2
    '''
    retImage1 = image1
    retImage2 = image2

    if image1.shape[2] is not 3 or image2.shape[2] is not 3:
        print("Image1 and Image2 were not the same color format. Converting to greyscale")
        if image1.shape[2] is 3:
            retImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        if image2.shape[2] is 3:
            retImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    return retImage1, retImage2
