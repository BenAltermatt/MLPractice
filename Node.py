import random

class Node: # This is a single 'neuron'
        # Default activation function
        def default_activation(self, x):
            return 1 if x >= .5 else 0

        # Makes a node with an input number of weights
        # or a passed array of weights
        def __init__(self, num_weights, weights = None, active_func = default_activation):
            self.__active_func = active_func

            # I am going to figure out whether
            # i need to generate some random weights
            # or just put a zero on there
            if weights is not None:
                self.__weights = weights
            else:
                self.__weights = list()
                for x in range(num_weights) : # generate random weights
                    self.__weights.append(random.random())
        
        # summs the weighted inputs and runs them through
        # the activation function
        def calculate(self, inputs):
            weighted_sum = 0
            for x in range((len(inputs))):
                weighted_sum += inputs[x] * self.__weights[x]

            return self.__active_func(weighted_sum)

        # this is just python's toStirng
        def __str__(self):
            holderString = "["
            for x in self.__weights:
                holderString += str(x) + ",\n"
            
            holderString += str(self.__active_func) + "]"
            return holderString