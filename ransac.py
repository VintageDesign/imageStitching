import numpy as np


def generateMatrix(points1,  points2):
    '''
    Generates a 2n x 9 matrix that the homography matrix can be derived from
    '''

    matrix = np.zeros((2 * len(points1), 9))

    for i in range(0, 2 * len(points1), 2):
        # Map the index from the size of the matrix, 8, to the number of points, 4.
        index = int(i/2)
        x  = points1[index][0]
        y  = points1[index][1]
        xp = points2[index][0]
        yp = points2[index][1]

        row1 = [ xp, yp , 1, 0, 0, 0, -x*xp, -x*yp, -x]
        row2 = [ 0, 0, 0, xp, yp, 1, -y*xp, -y*yp, -y]

        matrix[i,:] = row1[:]
        matrix[i+1,:] = row2[:]

    return matrix

def calcH(points1, points2):
    matrix = []
    for idx in range(0, len(points1)):
        x1 = points1[idx][0]
        y1 = points1[idx][1]
        x2 = points2[idx][0]
        y2 = points2[idx][1]

        r2 = [0, 0, 0, -1 * x1, -1 * y1, -1 * 1, y2 * x1, y2 * y1, y2 * 1]
        r1 = [-1 * x1, -1 * y1, -1 * 1, 0, 0, 0, x2 * x1, x2 * y1, x2 * 1]
        matrix.append(r1)
        matrix.append(r2)
    
        u, s, v = np.linalg.svd(matrix)

        h = np.reshape(v[8], (3,3))
        h = (1/h.item(8)) * h
    #print(matrix)
    #print(h)
    return h

def ransac(match1, match2):
    '''
    Implemenation of a RANSAC algorthm to find the a good approximation for the Homograpy
    '''
    fitness = 0
    iteration = 0

    while fitness < .8 :
        points = np.random.permutation(range(0,len(match1)))[:4]

        leftPoints = [match1[points[0]], match1[points[1]], match1[points[2]], match1[points[3]]]
        rightPoints = [match2[points[0]], match2[points[1]], match2[points[2]], match2[points[3]]]

        #matrix = generateMatrix(leftPoints, rightPoints)
        #U, theta, V = np.linalg.svd(matrix)
        #H = V[:, 8]
        #H = H / H[8]
        #print(H)
        #H = H.reshape((3,3))
        #print(H)
        
        H = calcH(leftPoints, rightPoints)
        fitPop = 0

        for i in range(0, len(match1)):
            # Coords must be 1x3 to multiply with the 3x3 Homography matrix
            coord = np.asarray([match1[i][0], match1[i][1], 1])
            expected = [match2[i][0], match2[i][1], 1]
            newcoord = np.matmul(H, coord.T)
            newcoord = newcoord / newcoord[2]
            #print(newcoord, expected)

            outcome = (newcoord[0] - expected[0],  newcoord[1] - expected[1])

            if -1 < outcome[0] < 1 and -1 < outcome[1] < 1:
                fitPop += 1
            #input()
        
        fitness = fitPop / len(match1)
        iteration += 1
        
    #print(H, iteration)
    print("Fitness percent: ",fitness, "  Matches: ", fitPop, "  Iterations: ", iteration)
    return H



