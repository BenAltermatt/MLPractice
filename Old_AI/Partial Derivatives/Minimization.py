def gv1(x, y):
    return (8 * x - 3 * y + 24, 4 * (y - 5) - 3 * x)


def gv2(x, y):
    return(200 * (x - y ** 2), 2 * y * (-200 * x + 200 * y ** 2 - 1))

def minimize(x, y, g_func, l):
    p = (x, y)
    gv = g_func(p[0], p[1])

    while abs(gv[0]) + abs(gv[1]) > 0:
        p = (p[0] + gv[0] * l, p[1] + gv[1] * l)
        gv = g_func(p[0], p[1])
        print(p)

    print('Local minimum found at: ' + str(p) + '\nValue of: ' + str())

minimize(0, 0, gv1, .0000005)