'''
Name: Benjamin Altermatt
Date: 3 - 31 - 2019

2 -
1 , 1
3

3 -
1 , 1 , 1
7

4 -
1 , 1 , 1, 1
15
'''
from matplotlib import pyplot as plt
import math
import random
import copy
from heapq import heappush, heappop

def perceptron(A, w, b, x):
    dot = 0
    for i in range(len(w)):
        dot += w[i] * x[i]
    return A(dot + b)

"""
This performs the calculation that a perceptron would perform from 
if it had:
    Activation function - A
    Weight vector       - w
    Bias value          - b
    Sigmoid coefficient - k
    Input vector        - x
"""
def new_percep (A, w, b, k, x):
    dot = 0
    for i in range(len(w)):
        dot += w[i] * x[i]
    return A(k, dot + b)

def step(x):
    if x > 0:
        return 1
    else:
        return 0

def sigmoid(k, z):
    return 1/ (1 + math.exp(-1 * k * z))

# Takes a # of bits and then the canonical representation of the truth table, then returns a truth table
def truth_table(bits, n):
    inputs = list()
    holder = str(bin(n)[2:])
    while(len(holder) < 2**bits):
        holder = '0' + holder
    pos_in = 0
    binary_val = 2**(bits) - 1
    temp = str(bin(binary_val))
    temp = temp[:temp.index('b')] + temp[temp.index('b') + 1:]
    temp = temp[-bits:]
    length = len(temp)
    for x in range(binary_val + 1):
        temp = str(bin(binary_val))
        temp = temp[:temp.index('b')] + temp[temp.index('b') + 1:]
        temp = temp[-bits:]
        vals = list()
        if(len(temp) < length):
            for x in range(length - len(temp)):
                vals.append(0)
        for x in temp:
            vals.append(int(x))
        vals = tuple(vals)
        inputs.append((vals, int(holder[pos_in])))

        pos_in += 1
        binary_val -= 1

    return tuple(inputs)

# Prints the truth table to not look like absolute trash (me)
def pretty_print_tt(table):
    header = 'Inputs'
    while(len(table[0][0]) * 2 > len(header)):
        header += ' '
    header += '|Outputs'
    print(header)
    holder = ''
    for x in range(len(header)):
        holder += '_'
    print(holder)
    indent = ''
    if len(table[0][0]) < 6:
        for x in range(6 - len(table[0][0]) * 2):
            indent += ' '
    for x in table:
        line = ''
        for y in x[0]:
            line += str(y) + ' '
        line += indent + '|   ' + str(x[1])
        print(line)

# Checks if my perceptron is wrong for a certain percentage of times
def check(n, w, b):
    table = truth_table(len(w), n)
    total = len(table)
    correct = 0
    for x in table:
        if perceptron(step, w, b, x[0]) == x[1]:
            correct += 1
    return correct / total

# Trains a perceptron to match a boolean function
def train(func, n):
    # Setting up a perceptron
    weight = list()
    b = 0
    for x in range(n):
        weight.append(1)

    #Set up table
    table = truth_table(n, func)

    for x in range(100): #All single perceptron logic gates of n < 4 can be trained in 100 epochs
        for x in table:
            y_star = perceptron(step, weight, b, x[0])
            err = x[1] - y_star
            for i in range(len(weight)):
                weight[i] = weight[i] + x[0][i] * err
                b  = b + err


    if check(func, weight, b) != 1:
        return False
    else:
        return tuple(weight), b

def chal_2():
    two = 0
    three = 0
    four = 0

    for x in range(16):
        if train(x, 2) is not False:
            two += 1

    print('2-Inputs Solvable: ' + str(two))


    for x in range(256):
        if train(x, 3) is not False:
            three += 1

    print('3-Inputs Solvable: ' + str(three))


    for x in range(65536):
        if train(x, 4) is not False:
            four += 1

    print('4-Inputs Solvable: ' + str(four))

def chal_3(A, w, b):
    coords = list()

    for x in range(1, 40):
        coords.append(-2 + x * .1)

    plt.axis([-2, 2, -2, 2])
    for x in coords:
        for y in coords:
            size = 2
            if (x == 1 or x == 0) and (y == 1 or y == 0):
                size = 5
            if perceptron(A, w, b, [x, y]) == 1:
                plt.plot([x], [y], 'bo', markersize = size)
            else:
                plt.plot([x], [y], 'ro', markersize = size)


    plt.show()

def network_4(x, y):
    return perceptron(step, [2, 3], -4, [perceptron(step, [-1,-3], 4, [x, y]), perceptron(step, [1, 1], 0, [x, y])])


def chal_4():
    coords = list()

    for x in range(1, 40):
        coords.append(-2 + x * .1)

    plt.axis([-2, 2, -2, 2])
    for x in coords:
        for y in coords:
            size = 2
            if (x == 1 or x == 0) and (y == 1 or y == 0):
                size = 5
            if network_4(x, y) == 1:
                plt.plot([x], [y], 'bo', markersize=size)
            else:
                plt.plot([x], [y], 'ro', markersize=size)

    plt.show()

def threshold_step(x, threshold):
    if x > threshold:
        return 1
    else:
        return 0

# This essentially will take tuples of different values corresponding to different perceptrons and output the output of the network
"""
This actually shows a lot about how I modeled these perceptrons.

You can see here that I used 5 in my model. the 0 1 2 and 3 were in the hidden
layer, which is why they only had two weights (connecting them and the input x and y coords).

There is a better way to do this, I think.
"""
def run_circ_brain(weights, b_vals, k_val, threshold, input):
    # First layer
    percep_0 = new_percep(sigmoid, [weights[0], weights[1]], b_vals[0], k_val, [input[0], input[1]])
    percep_1 = new_percep(sigmoid, [weights[2], weights[3]], b_vals[1], k_val, [input[0], input[1]])
    percep_2 = new_percep(sigmoid, [weights[4], weights[5]], b_vals[2], k_val, [input[0], input[1]])
    percep_3 = new_percep(sigmoid, [weights[6], weights[7]], b_vals[3], k_val, [input[0], input[1]])
    percep_4 = new_percep(sigmoid, [weights[8], weights[9], weights[10], weights[11]], b_vals[4], k_val, [percep_0, percep_1, percep_2, percep_3])

    if percep_4 > threshold:
        return 1
    else:
        return 0

# Returns whether a coordinate is actually within the unit circle
def is_in_circ(x, y):
    return 1 > (x * x + y * y)

# Creates the first generation of brains
"""
I called networks 'brains' in high school, I guess. 
What they _really_ were was tuples that represented 12 weights
and 5 bias values.

Thats cuz my network looked something liked

        p1
xin     p2      p5
        p3
yin     p4

where there were edges or "weights" between the inputs
and each middle layer perceptron, and then each 
perceptron in the middle layer and last perceptron
"""
def gen_first_gen():
    brains = list()
    # list of layers

    # for 20 layers
    for x in range(20):
        weights = list()
        # make a list of 12 weights (what)
        for x in range(12):
            weights.append(1)

        # make a list of 5 biases (huh)
        b_vals = list()
        for x in range(5):
            b_vals.append(0)

        # I made these in to tuples because you can hash them
        # they are actually based and redpilled
        brains.append((tuple(weights), tuple(b_vals)))

    return tuple(brains)

# Generates a mutated generation from the best half of the old gen
"""
This is what I used to approximate the idea of 
some kind of machine learning aglorithm before I knew
anything about gradient descent.

I selected the best of the old generation,
and randomly altered the values by some specified
range that depended on how accurate I already was.
"""
def gen_new(old, mut_mag, mut_num):
    new_gen = list()

    for x in old: # For each old one
        for i in range(2): # Create two descendants
            weights = list(x[0])

            # Num we're altering
            num_altered = random.randint(0, int(mut_num))
            for y in range(num_altered):
                temp = random.choice(range(len(weights)))
                weights[temp] = weights[temp] + (random.randint(int(-100 * mut_mag), int(100 * mut_mag)) / 100)

            b_vals = list(x[1])

            num_altered = random.randint(0, int(mut_num  / 12 * 5))
            for y in range(num_altered):
                temp = random.choice(range(len(b_vals)))
                b_vals[temp] = b_vals[temp] + (random.randint(int(-100 * mut_mag),int( 100 * mut_mag)) / 100)
            new_gen.append((tuple(weights), tuple(b_vals)))

    return tuple(new_gen)

# Tests the accuracy of a brain
def test_accuracy(brain, k_val, threshold):
    coords = list()

    for x in range(1, 40):
        coords.append(-2 + x * .1)

    # plt.axis([-2, 2, -2, 2])
    correct = 0
    total = 0
    for x in coords:
        for y in coords:
            total += 1
            if is_in_circ(x, y) == run_circ_brain(brain[0], brain[1], k_val, threshold, [x, y]):
                correct += 1
            #     plt.plot([x], [y], 'bo')
            # else:
            #     plt.plot([x], [y], 'ro')

    # plt.show()

    return correct / total


# Tests the fitness of a generation
def test_fitness(generation, k_val, threshold):
    ranks = list()

    for x in generation:
        heappush(ranks, (1 - test_accuracy(x, k_val, threshold), x))

    top_survivors = list()
    total_fitness = 0
    for x in range(len(generation) // 2):
        temp = heappop(ranks)
        top_survivors.append(temp[1])
        total_fitness += 1 - temp[0]
        # print('Gen len: ' + str(len(generation)))
    return total_fitness / (len(generation) / 2), tuple(top_survivors)


def volatility_vals(init_mag, init_num, accuracy):
    mut_mag = -5 * init_mag / 4 * accuracy + 5 * init_mag / 4
    mut_num = -5 * init_num / 4 * accuracy + 5 * init_num / 4
    return mut_mag, mut_num


def show_me(brain, k_val, threshold):
    coords = list()

    for x in range(1, 40):
        coords.append(-2 + x * .1)

    plt.axis([-2, 2, -2, 2])
    for x in coords:
        for y in coords:
            if is_in_circ(x, y):
                if run_circ_brain(brain[0], brain[1], k_val, threshold, [x, y]):
                    plt.plot([x], [y], 'bo')
                else:
                    plt.plot([x], [y], 'co')
            else:
                if run_circ_brain(brain[0], brain[1], k_val, threshold, [x, y]):
                    plt.plot([x], [y], 'mo')
                else:
                    plt.plot([x], [y], 'wo')

    plt.show()


def chal_5():
    current_gen = gen_first_gen()

    # Constants
    k_val = 1
    threshold = .5
    init_mag = 30
    init_num = 60

    gen_num = 0
    show_me(current_gen[0], k_val, threshold)
    darwined = test_fitness(current_gen, k_val, threshold)
    avg_acc = darwined[0]
    current_gen = darwined[1]

    while avg_acc < .99:
        gen_num += 1
        print('Generation ' + str(gen_num) + ' accuracy: ' + str(100 * avg_acc) + '%')
        print('Best values: ' + str(current_gen[0]))

        if gen_num % 100 == 0:
            show_me(current_gen[0], k_val, threshold)

        volatility = volatility_vals(init_mag, init_num, avg_acc)
        current_gen = gen_new(current_gen, volatility[0], volatility[1])
        # print('gen len: ' + str(len(current_gen)))
        darwined = test_fitness(current_gen, k_val, threshold)
        avg_acc = darwined[0]
        current_gen = darwined[1]
        # print('gen len2: ' + str(len(current_gen)))

chal_5()
