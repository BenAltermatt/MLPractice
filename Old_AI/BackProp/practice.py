import numpy as np

x = np.matrix([[1, 2],
               [3, 4]])
y = np.matrix([[2, 3]])

# print(x + y)
print(y * x)
# print(np.multiply(x, y))
# print(4*x)
# print(x.transpose())
# print(x[0, 1])
# print(y[1, 1])
#
# def ex_f(x):
#     return x**2 + 1
#
# vec_f = np.vectorize(ex_f)
#
# print(vec_f(x))
#
# print(np.linalg.norm(np.matrix([[1, -1]])))
# print(np.linalg.norm(np.matrix([[-3, 4]])))
