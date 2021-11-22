import random
import Node

class NeuralNetwork: # a neural network made of nodes
    # for now, it will be a traditional, fully connected network
    __num_inputs = 0
    __num_outputs = 0
    __normalize = False
    __layers = None
    __output_layer = None

    # finds the number of input and output values and
    # whether the inputs should be normalized  between 0 and 1
    def __init__(self, num_inputs=1, num_outputs=1, normalize=False):
        global __num_inputs
        global __num_outputs
        global __normalize
        global __layers 
        global __output_layer

        __num_inputs = num_inputs
        __num_outputs = num_outputs
        __normalize = True
        __layers = list()
        __output_layer = self.add_output()

    # create an output_layer
    def add_output(self, activation_function=None):
        final_layer = list()
        # we need to know the number of inputs
        # for the input node
        num_inputs = __num_inputs

        if len(__layers) > 0: # we have a middle layer
            num_inputs = len(__layers[-1])
        
        for x in range(__num_outputs):
            if activation_function is not None:
                final_layer.append(Node(num_inputs, active_func=activation_function))
            else:
                final_layer.append(Node(num_inputs))
            
        return final_layer

    # adds a layer of n specified nodes at the designated index
    # with a specified activation functiopn
    def add_layer(self, n, index=-1, activation_func=None):
        node_group = list()
        num_weights = 0

        # make it set to the largest
        if index == -1:
            index = len(__layers)

        # find out if this is the first layer
        if len(__layers) == 0:
            num_weights = __num_inputs
        else:
            num_weights = len(__layers[-1])

        # build each node and add it to the layer
        for x in range(n) :
            if activation_func is not None:
                node_group.append(Node(num_weights, activ_func=activation_func))
            else:
                node_group.append(Node(num_weights))

        self.add_output()