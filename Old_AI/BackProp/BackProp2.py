'''
The Bs Lists and the Dot lists are not zero indexed. You have to append None to them initially for it to work out.
'''

import numpy as np
import math, random

#HELPER METHODS

def propagate(input, output, weights, bs, act, dact, lmd, num):
    N = len(weights)

    a = list()
    dot = list()
    a.append(input)
    dot.append(None)

    for L in range(1, N + 1):
        dot.append(a[L - 1]*weights[L - 1] + bs[L])
        a.append(np.vectorize(act)(dot[L]))

    # Print error
    err = .5 * (np.linalg.norm(output - a[len(a) - 1])) ** 2
    print('Num: ' + str(num))
    print('Error: ' + str(err) + '\n')
    # print('Dots: ' + str(dot))
    # print('As: ' + str(a))

    #Make a list of the delta values
    delta = list()
    for x in range(N):
        delta.append(None)

    #Add dN
    delta.append(np.multiply(np.vectorize(dact)(dot[N]), output - a[N]))

    for L in range(N - 1, 0, -1):
        delta[L] = np.multiply(np.vectorize(dact)(dot[L]), delta[L + 1] * weights[L].transpose())

    # print('weights: ' + str(weights))
    # print('bs: ' + str(bs))
    # print('∆: ' + str(delta))

    #Alter the weights
    for L in range(0, N):
        weights[L] = weights[L] + lmd * a[L].transpose()*(delta[L + 1])
        bs[L + 1] = bs[L + 1] + lmd * delta[L + 1]

    return err, weights, bs


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def deriv_sigmoid(x):
    return math.exp(-x)/(1 + math.exp(-x)) ** 2

def discreet(x):
    return int(x > 0)

def forward_prop(input, weights, bs, act):
    N = len(weights)
    outputs = list()

    for x in input:
        a = list()
        dot = list()
        a.append(x)
        dot.append(None)

        for L in range(1, N + 1):
            dot.append(a[L - 1] * weights[L - 1] + bs[L])
            a.append(np.vectorize(act)(dot[L]))

        outputs.append(a[len(a) - 1])

    for x in range(len(input)):
        print('Input: ' + str(input[x]) +'\nOutput: ' + str(outputs[x]) + '\n')

    return outputs


def back_prop(training_set, lmd, act, dact, architecture):
    while not False:

        weights = list()
        bs = list()
        bs.append(None)

        for x in range(len(architecture)):
            # Add a weight if needed
            if x < len(architecture) - 1:
                weights.append(np.random.rand(x, x+1))

            # Add a b of needed
            if x > 0:
                bs.append(np.random.rand(1, x))

        epoch = 0
        while True:
            epoch += 1
            for x in range(len(training_set)):
                temp = propagate(training_set[x][0], training_set[x][1], weights, bs, act, dact, lmd, x)



# TESTING METHODS
def part_1():
    print('\nPart 1')
    #Testing inputs 1
    x = np.matrix([[2, 3]])
    y = np.matrix([[.8, 1]])

    #Testing weights 1
    weights = list()

    weights.append(np.matrix([[-1, -.5],
                    [1, .5]]))
    weights.append(np.matrix([[1, 2],
                              [-1, -2]]))

    #Testing bs 1
    bs = list()
    bs.append(None)
    bs.append(np.matrix([[1, -1]]))
    bs.append(np.matrix([[-.5, .5]]))

    for i in range(2):
        propagate(x, y, weights, bs, sigmoid, deriv_sigmoid, .1)


def part_2():
    print('\nPart 2')
    x = list()
    x.append(np.matrix([[1, 1]]))
    x.append(np.matrix([[1, 0]]))
    x.append(np.matrix([[0, 1]]))
    x.append(np.matrix([[0, 0]]))

    weights = list()
    weights.append(np.matrix([[-3, 1],
                              [-1, 1]]))
    weights.append(np.matrix([[2],
                   [3]]))

    bs = list()
    bs.append(None)
    bs.append(np.matrix([[4, 0]]))
    bs.append(np.matrix([[-4]]))

    forward_prop(x, weights, bs, discreet)

def part_3():
    print('\nPart 3')

    training_set = list()

    training_set.append((np.matrix([[0, 0]]),np.matrix([[0, 0]])))
    training_set.append((np.matrix([[0, 1]]),np.matrix([[0, 1]])))
    training_set.append((np.matrix([[1, 0]]),np.matrix([[0, 1]])))
    training_set.append((np.matrix([[1, 1]]),np.matrix([[1, 0]])))

    holder = back_prop(training_set, 10, sigmoid, deriv_sigmoid, [2, 2])

    input = list()
    for x in range(training_set):
        input.append(x[0])

    temp = forward_prop(input, holder[0], holder[1], sigmoid)



# part_1()
# part_2()
part_3()