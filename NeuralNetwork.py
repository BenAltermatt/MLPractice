import random
from Node import Node

class NeuralNetwork: # a neural network made of nodes
    # for now, it will be a traditional, fully connected network

    # finds the number of input and output values and
    # whether the inputs should be normalized  between 0 and 1
    def __init__(self, num_inputs=1, num_outputs=1, normalize=False):
        self.__num_inputs = num_inputs
        self.__num_outputs = num_outputs
        self.__normalize = normalize
        self.__layers = list()
        self.add_output()

    # create an output_layer
    def add_output(self, activation_function=None):
        final_layer = list()
        # we need to know the number of inputs
        # for the input node
        num_inputs = self.__num_inputs

        if len(self.__layers) > 0: # we have a middle layer
            num_inputs = len(self.__layers[-1])
        
        for x in range(self.__num_outputs):
            if activation_function is not None:
                final_layer.append(Node(num_inputs, active_func=activation_function))
            else:
                final_layer.append(Node(num_inputs))

        self.__output_layer = final_layer

    # adds a layer of n specified nodes at the designated index
    # with a specified activation functiopn
    def add_layer(self, n, index=-1, activation_func=None):
        node_group = list()
        num_weights = 0

        # make it set to the largest
        if index == -1:
            index = len(self.__layers)

        # find out if this is the first layer
        if len(self.__layers) == 0:
            num_weights = self.__num_inputs
        else:
            num_weights = len(self.__layers[-1])

        # build each node and add it to the layer
        for x in range(n) :
            if activation_func is not None:
                node_group.append(Node(num_weights, activ_func=activation_func))
            else:
                node_group.append(Node(num_weights))

        self.__layers.insert(index, node_group)

        self.add_output()

    # we need a string representation of this for
    # better testing
    def __str__(self):
        retStr = ""
        for x in range(len(self.__layers)):
            retStr += "\nLayer " + str(x) + "\n========\n"
            print(retStr)
            for y in self.__layers[x]:
                retStr += str(y) + "\n"
            retStr += "========\n"
        # now add the output layer
        retStr += "\nOutput Layer\n========\n"
        for x in self.__output_layer:
            retStr += str(x)
        retStr += "\n========\n"
        return retStr

    # calculate by running through the algorithm
    def calculate(self, inputs):
        outputs = inputs
        ins = None
        for x in self.__layers: # for each layer of nodes
            ins = outputs
            outputs = []
            for y in x: # each node in the layer
                outputs.append(y.calculate(ins))

        # go through the last layer
        final_outs = []
        for x in self.__output_layer:
            final_outs.append(x.calculate(outputs))
        
        return tuple(final_outs)

    # finds the max values for every part of a state
    # tecnically every component of an input vector
    def find_max(self, vals):
        maxs = list(vals[0])

        for x in vals:
            for y in range(len(x)):
                maxs[y] = max(maxs[y], x[y])
        
        return maxs
    
    # finds the min values for every part of a state
    # technically every component of an input vector
    def find_min(self, vals):
        mins = list(vals[0])
        for x in vals:
            for y in range(len(x)):
                mins[y] = min(mins[y], x[y])

        return mins

    # Will find the values that are the min and maxs
    # and normalize them to the range 0 < x < 1
    def normalize(self, values):
        maxs = self.find_max(values)
        mins = self.find_min(values)

        new_inputs = list()
        for x in values:
            new_input = list()
            for y in range(len(x)):
                new_input.append((x[y] - mins[y]) / (maxs[y] - mins[y]))
            new_inputs.append(new_input)
        
        return new_inputs

    # calculate for all inputs
    def calculate_all(self, inputs):
        ret_vals = list()

        if self.__normalize is True:
            inputs = self.normalize(inputs)

        for x in inputs:
            ret_vals.append(self.calculate(x))
        
        return tuple(ret_vals)