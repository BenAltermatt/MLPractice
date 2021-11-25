import numpy as np
import random, math

from Old_AI.HillClimb.HillClimb import sigmoid

"""
I have no clue what the hell I did in high school this shit
is crazy. I am going to relearn it. 
"""

"""
I'll start off by trying to hard code a 2 - 4 - 1 network
"""

# returns a tuple representing a network based
# on a passed structure
def gen_brain(structure, w_min, w_max, b_min, b_max, k_min, k_max):
    layers = list()

    # generate the layers
    for x in range(1, len(structure)):

        # generate the weight and bias matrices for this layer
        weights = list()
        biases = list()
        sigmoid_constants = list()
        for x in range(structure[x]):
            # generate the weight vector for this perceptron
            w_vec = list()
            for y in range(structure[x - 1]):
                w_vec.append(random.random() * (w_max - w_min) + w_min)
            weights.append(tuple(w_vec))

            # generate a bias for this perceptron
            biases.append(random.random() * (b_max - b_min) + b_min)

            #generate a sigmoid constant list for this perceptron
            sigmoid_constants.append(random.random() * (k_max - k_min) + k_max)

        # make the weights matrix
        w_matrix = np.matrix(tuple(weights))

        # make the bias matrix
        b_matrix = np.matrix(tuple(biases,))

        # combine all info into a tuple representing a layer
        layers.append(tuple(w_matrix, b_matrix, tuple(sigmoid_constants)))
    
    return tuple(layers)

def print_brain(brain):
    print(brain)


test_brain = gen_brain([2, 4, 1], -1, 1, -1, 1, 0, 5)
print_brain(test_brain)
        
        

