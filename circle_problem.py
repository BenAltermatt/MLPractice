import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import NeuralNetwork

def binary_func(x):
    return 1 if x >= .5 else 0

network = NeuralNetwork(num_inputs=2, normalize=True, output_func=binary_func)
network.add_layer(2)
network.add_layer(4)

def generate_points(x_s, y_s, w, h, i):
    points = list()
    for x in np.arange(x_s, w, i):
        for y in np.arange(y_s, h, i):
            points.append((x, y))
    
    return points

def plot_circle(x_c,y_c,r):
    points = generate_points(x_c - r, y_c - r, r * 2, r * 2, .025)
    
    solutions = network.calculate_all(points)
    accuracy = 0.0
    # calculate using the NN and compare it to accurate
    for x in range(len(points)):       
        print(solutions[x][0])

        if solutions[x][0] > .5 and (points[x][0] - x_c)**2 + (points[x][1] - y_c)**2 < r**2:
            accuracy += 1
        if solutions[x][0] <=.5 and (points[x][0] - x_c)**2 + (points[x][1] - y_c)**2 >= r**2:
            accuracy += 1

        if((points[x][0] - x_c)**2 + (points[x][1] - y_c)**2 < r**2):
            if solutions[x][0] > .5:
                plt.plot(points[x][0], points[x][1], 'o', color='orange')
            else:
                plt.plot(points[x][0], points[x][1], 'ro')
        else:
            if solutions[x][0] > .5:
                plt.plot(points[x][0], points[x][1], 'go')
            else:
                plt.plot(points[x][0], points[x][1], 'bo')
                
    plt.axis([0, 1, 0, 1])
    plt.show()

def main():
    plot_circle(.5, .5, .5)
    return

if __name__ == '__main__':
    main()

