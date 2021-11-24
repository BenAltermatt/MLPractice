import math

def one_d_minimize(func, left, right, tol):
    if right - left < tol:
        return (left + right) / 2

    one3 = left + (right - left) / 3
    two3 = left + 2 * (right - left) / 3

    if func(two3) > func(one3):
        return one_d_minimize(func, left, two3, tol)
    else:
        return one_d_minimize(func, one3, right, tol)


def mag_func(f, pos, grad):
    def func(lmda):
        return f(subtract_vecs(pos, scale_vec(lmda, grad)))
    return func


def gradient_descent(f, gvf, start, tol):
    loc = start
    while math.sqrt(gvf(loc)[0] ** 2 + gvf(loc)[1] ** 2) > tol:
        dir = gvf(loc)
        change_val = mag_func(f, loc, dir)
        lmda = one_d_minimize(change_val, 0, 1, 10**(-8))
        # print('Pre Len: ' + str(len(loc)))
        loc = subtract_vecs(loc, scale_vec(lmda, dir))
        # print('Post Len: ' + str(len(loc)) + '\n')
    return loc


#HELPER METHODS
def add_vecs(v1, v2):
    temp = list()

    for x in range(len(v1)):
        temp.append(v1[x] + v2[x])
    return tuple(temp)

def mult_vecs(v1, v2):
    temp = list()
    for x in range(len(v1)):
        temp.append(v1[x] * v2[x])
    return tuple(temp)

def scale_vec(x, v1):
    temp = list()
    for i in v1:
        temp.append(x * i)
    return tuple(temp)

def subtract_vecs(v1, v2):
    v2 = scale_vec(-1, v2)
    return add_vecs(v1, v2)


#TESTING METHODS
def func_1(v):
    return 4 * v[0] ** 2 - 3 * v[0] * v[1] + 2 * v[1] ** 2 + 24 * v[0] - 20 * v[1]

def gv1(v):
    return(8 * v[0] - 3 * v[1] + 24, 4 * (v[1] - 5) - 3 * v[0])

def func_2(v):
    return((1 - v[1]) ** 2 + 100* (v[0] - v[1] ** 2) ** 2)

def gv2(v):
    return (200 * (v[0] - v[1] ** 2)),(2 * (-200 * v[0] * v[1] + 200 * v[1] ** 3 + v[1] - 1))

print(gradient_descent(func_1, gv1, (0, 0), .0001))
print(gradient_descent(func_2, gv2, (0, 0), .0001))