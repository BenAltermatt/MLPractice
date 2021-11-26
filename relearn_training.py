import numpy as np
import random, math

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
        for y in range(structure[x]):
            # generate the weight vector for this perceptron
            w_vec = list()
            for z in range(structure[x - 1]):
                w_vec.append(random.random() * (w_max - w_min) + w_min)
            weights.append(tuple(w_vec))

            # generate a bias for this perceptron
            biases.append(random.random() * (b_max - b_min) + b_min)

            #generate a sigmoid constant list for this perceptron
            sigmoid_constants.append(random.random() * (k_max - k_min) + k_min)

        # make the weights matrix
        w_matrix = np.matrix(tuple(weights))

        # make the bias matrix
        b_matrix = np.matrix(tuple(biases)).transpose()

        # combine all info into a tuple representing a layer
        layers.append((w_matrix, b_matrix, tuple(sigmoid_constants)))
        
    return tuple(layers)

# this will forward propagate through the network and get the results
def run_network(input_vals, network, activation):

    output = np.matrix(input_vals).transpose()

    # goes through each layer
    for x in network:
        ys = x[0]* output + x[1] # Y= MX + B
        # now we need to perform the activation function to get the outputs
        activs = list()
        ys = ys.tolist()
        for y in range(len(ys)):
            activs.append(activation(ys[y][0], x[2][y])) # pass the k value for the layer
        output = np.matrix(activs).transpose()
    
    # evetnually we shopuld get through all of our
    # layers and have the final output
    return output.tolist()[0]

# sigmoid function used as a basic activation
def sigmoid(x, k):
    return 1 / (1 + math.exp(-k * x))

# this will print out the brain
def print_brain(brain):
    print(brain)


test_brain = gen_brain([2, 4, 1], -1, 1, -1, 1, 0, 5)
print(run_network((0, 0), test_brain, sigmoid))
        
