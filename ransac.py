""" This module performs the RANSAC algorithm using our homography matrix
    function in order to coorelate points between two lists """
import sys
sys.path.append("./depend/")
import numpy as np

def calc_h(points1, points2):
    """
        Calculates the homography matrix using two lists of corresponding points
    """
    matrix = [] # Create a blank matrix to use
    for idx, _ in enumerate(points1, start=0): #enumerate so we have an index value to use
        x_1 = points1[idx][0]
        y_1 = points1[idx][1]
        x_2 = points2[idx][0]
        y_2 = points2[idx][1]

        r_2 = [0, 0, 0, -1 * x_1, -1 * y_1, -1 * 1, y_2 * x_1, y_2 * y_1, y_2 * 1] # Create row 2
        r_1 = [-1 * x_1, -1 * y_1, -1 * 1, 0, 0, 0, x_2 * x_1, x_2 * y_1, x_2 * 1] # Create row 1
        matrix.append(r_1) # Add row 1 to matrix
        matrix.append(r_2) # Add row 2 to matrix
    _, _, v_val = np.linalg.svd(matrix) #U, S, V, We only need the V matrix from the decomp

    h_mat = np.reshape(v_val[8], (3, 3)) # Turn the V matrix into an actual matrix from an array
    h_mat = (1 / h_mat.item(8)) * h_mat # Normalize the h_matrix
    #print(matrix)
    #print(h)
    return h_mat

def ransac(match1, match2, CUTOFF = .8, MAX_RANSAC = 1000):
    '''
    Implemenation of a RANSAC algorthm to find the a good approximation for the Homograpy
    '''
    bestfit = 0
    bestmatches = 0
    best_h = np.reshape([1, 1, 1, 1, 1, 1, 1, 1, 1], (3,3)) # all 1's should prevent any translation if this is selected
    fitness = 0
    iteration = 0
    # Run until the max number of iterations or if we have a perfect match 1.0
    while iteration < MAX_RANSAC and bestfit != 1.0:
        points = np.random.permutation(range(0, len(match1)))[:4] # Get a random set of 4 matches from our match lists
        #Separate the points by image
        left_points = [match1[points[0]], match1[points[1]], match1[points[2]], match1[points[3]]] 
        right_points = [match2[points[0]], match2[points[1]], match2[points[2]], match2[points[3]]]
        h_mat = calc_h(left_points, right_points) # Calculate the h matrix
        fit_pop = 0 # zero out the fit percent

        # Run through the entire list of matches
        for i, _ in enumerate(match1, start=0):
            # Coords must be 1x3 to multiply with the 3x3 Homography matrix
            coord = np.asarray([match1[i][0], match1[i][1], 1]) # Create an array for the point (x, y, 1)
            expected = [match2[i][0], match2[i][1], 1] # Create an array for the expected result
            new_coord = np.matmul(h_mat, coord.T) # Do the multiplication of the point * homography matrix
            new_coord = new_coord / new_coord[2] # Normalize the result by dividing (x, y, z) by z
            #print(new_coord, expected)

            outcome = (new_coord[0] - expected[0], new_coord[1] - expected[1]) # Check how close together the new point and expected point are

            #If they are within 1 pixel count it as a match
            if (-1 < outcome[0] < 1) and (-1 < outcome[1] < 1): 
                fit_pop += 1
            # input()
        fitness = fit_pop / len(match1) # Calculate total number of match percent
        iteration += 1 # Inc iteration counter
        if fitness > bestfit: # Check if this is now the best match
            # print("Best fit: " + str(fitness))
            best_h = h_mat   # Save results
            bestfit = fitness
            bestmatches = fit_pop
    
    print("\tFitness percent: ", bestfit, "  Matches: ", bestmatches, " of ", len(match1) ,"  Iterations: ", iteration)
    return best_h
