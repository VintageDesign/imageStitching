""" This module performs the RANSAC algorithm using our homography matrix
    function in order to coorelate points between two lists """
import sys
sys.path.append("./depend/")
import numpy as np

def calc_h(points1, points2):
    """
        Calculates the homography matrix using two lists of corresponding points
    """
    matrix = []
    for idx, _ in enumerate(points1, start=0):
    #for idx in range(0, len(points1)):
        x_1 = points1[idx][0]
        y_1 = points1[idx][1]
        x_2 = points2[idx][0]
        y_2 = points2[idx][1]

        r_2 = [0, 0, 0, -1 * x_1, -1 * y_1, -1 * 1, y_2 * x_1, y_2 * y_1, y_2 * 1]
        r_1 = [-1 * x_1, -1 * y_1, -1 * 1, 0, 0, 0, x_2 * x_1, x_2 * y_1, x_2 * 1]
        matrix.append(r_1)
        matrix.append(r_2)
    _, _, v_val = np.linalg.svd(matrix) #U, S, V

    h_mat = np.reshape(v_val[8], (3, 3))
    h_mat = (1 / h_mat.item(8)) * h_mat
    #print(matrix)
    #print(h)
    return h_mat

def ransac(match1, match2, CUTOFF = .8, MAX_RANSAC = 100000):
    '''
    Implemenation of a RANSAC algorthm to find the a good approximation for the Homograpy
    '''
    fitness = 0
    iteration = 0
    while fitness < CUTOFF and iteration < MAX_RANSAC:
        points = np.random.permutation(range(0, len(match1)))[:4]

        left_points = [match1[points[0]], match1[points[1]], match1[points[2]], match1[points[3]]]
        right_points = [match2[points[0]], match2[points[1]], match2[points[2]], match2[points[3]]]
        h_mat = calc_h(left_points, right_points)
        fit_pop = 0

        for i, _ in enumerate(match1, start=0):
        #for i in range(0, len(match1)):
            # Coords must be 1x3 to multiply with the 3x3 Homography matrix
            coord = np.asarray([match1[i][0], match1[i][1], 1])
            expected = [match2[i][0], match2[i][1], 1]
            new_coord = np.matmul(h_mat, coord.T)
            new_coord = new_coord / new_coord[2]
            #print(new_coord, expected)

            outcome = (new_coord[0] - expected[0], new_coord[1] - expected[1])

            if (-1 < outcome[0] < 1) and (-1 < outcome[1] < 1):
                fit_pop += 1
            # input()
        fitness = fit_pop / len(match1)
        iteration += 1
    
    print("\tFitness percent: ", fitness, "  Matches: ", fit_pop, "  Iterations: ", iteration)
    return h_mat
