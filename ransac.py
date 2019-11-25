def generateMatrix(points1,  points2):
    '''
    Generates a 2n x 9 matrix that the homography matrix can be derived from
    '''

    matrix = np.zeros((2 * len(points1), 9))

    for i in range(0, len(points1):
            x  = points1[i][0]
            y  = points1[i][1]
            xp = points2[i][0]
            yp = points2[i][1]

            row1 = [ xp, ypx , 1, 0, 0, 0, -x*xp, -x*yp, -x]
            row2 = [ 0, 0, 0, xp, yp, 1, -y*xp, -y*yp, -y]





def ransac(match1, match2):
    '''
    Implemenation of a RANSAC algorthm to find the a good approximation for the Homograpy
    '''



    fitness = 0

    while fitness < .8:
        points = np.random.permutation(range(0,len(match1)))[:4]

        leftPoints = [match1[points[0]], match1[points[1]], match1[points[2]], match1[points[3]]]
        rightPoints = [match2[points[0]], match2[points[1]], match2[points[2]], match2[points[3]]]


        matrix = generateMatrix(leftPoints, rightPoints)


