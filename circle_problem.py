import matplotlib.pyplot as plt
import numpy as np

def generate_points(x_s, y_s, w, h, i):
    points = list()
    for x in np.arange(x_s, w, i):
        for y in np.arange(y_s, h, i):
            points.append((x, y))
    
    return points

def plot_circle(x_c,y_c,r):
    points = generate_points(x_c - r, y_c - r, r * 2, r * 2, .025)
    for point in points:        
        if((point[0] - x_c)**2 + (point[1] - y_c)**2 < r**2):
            plt.plot(point[0], point[1], 'ro')
        else:
            plt.plot(point[0], point[1], 'bo')
        
    
    plt.axis([0, 1, 0, 1])
    plt.show()
    """
    plt.plot(.5, .5, 'ro')
    plt.plot(.6,.5,'bo')
    plt.axis([0, 1, 0, 1])
    plt.show()
    """

def main():
    plot_circle(.5, .5, .5)
    return

if __name__ == '__main__':
    main()

