# test af exp precision

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def myExp(x, l):
    '''
    My implementation of a exp function

        x^n
    sum ---
        n!

    x is input value
    l is the number of joints for the approximation

    return e^x
    '''
    e = 0
    for n in range(l):
        e = e + ((x**n) / (math.factorial(n)))

    return e


def myExpV(x, l):
    '''
    My implementation of a exp function

        x^n
    sum ---
        n!

    x is input value
    l is the number of joints for the approximation

    return e^x
    '''
    e = 0
    for n in range(l):
        p = (x**n)
        f = (math.factorial(n))
        led = (p / f)
        e = e + led
        print n, ': the exponent is:', e, 'p is', p, 'f is', f, 'led is', led

    return e


def soExp(x):
    e = 0
    pres = 2**(-8)  # lowest value for a 8 bit fixed point value
    s = 1
    i = 1
    while np.abs(s) > pres:
        e = e + s
        s = (x**i) / math.factorial(i)
        i = i + 1
    return e, i


if __name__ == '__main__' or True:

    plt.close("all")
    L = 6
    sig = 25.0
    inputs = -np.arange(256) / sig  # 255.0 / sig
    # print inputs

    # make a 3d plot
    # Start with variable for output. data[L][x]
    data = np.zeros((50, 256))
    mExp = np.zeros((50, 256))
    realExp = np.zeros((50, 256))
    L = np.arange(1, 51)
    # print L.shape, data.shape, inputs.shape
    maxVal = 1
    maxVal2 = 1.01
    for l in range(50):
        for x in range(len(inputs)):
            if np.abs(myExp(inputs[x], l) - np.exp(inputs[x])) < maxVal:
                data[l, x] = np.abs(myExp(inputs[x], l) - np.exp(inputs[x]))
            else:
                data[l, x] = maxVal
            realExp[l, x] = np.exp(inputs[x])
            if myExp(inputs[x], l) < maxVal2 and \
                    myExp(inputs[x], l) > -maxVal2:
                mExp[l, x] = myExp(inputs[x], l)
            elif myExp(inputs[x], l) > maxVal2:
                mExp[l, x] = maxVal2
            else:
                mExp[l, x] = -maxVal2

    # print inputs[255], soExp(inputs[255])
    # print myExpV(inputs[-1], 35), np.exp(inputs[-1])

    # calculate values for x**n and factorial comparison
    clen = 100
    p = np.zeros(clen)
    f = np.zeros(clen)
    for i in range(clen):
        p[i] = 10**i
        f[i] = math.factorial(i)

    # figure 1 ------------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Iv, Lv = np.meshgrid(inputs, L)

    Z = data  # np.log10(data)
    surf = ax.plot_surface(Lv, Iv, Z, rstride=2, cstride=2,
                           cmap=cm.coolwarm, linewidth=0.5, antialiased=True)

    ax.set_zlim(0, maxVal)
    ax.set_xlabel('Number of terms')
    ax.set_ylabel('x in exp(x)')
    ax.set_zlabel('Error')
    ax.view_init(40, 45)
    fstr = 'data/exp_err_surf.png'
    plt.savefig(fstr)

    # figure 2 ----------------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(Lv, Iv, Z, rstride=5, cstride=15)

    ax.set_zlim(0, maxVal)
    ax.set_xlabel('Number of terms')
    ax.set_ylabel('x in exp(x)')
    ax.set_zlabel('Error')
    ax.view_init(30, 90)
    fstr = 'data/exp_err_wire.png'
    plt.savefig(fstr)

    # figure 3 ----------------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(Lv, Iv, realExp, rstride=5, cstride=15, label='Numpy')
    ax.plot_wireframe(Lv, Iv, mExp, rstride=1, cstride=10, color="red",
                      label='Estimate')

    ax.set_zlim(-maxVal2, maxVal2)
    ax.set_ylim(inputs[255], 0)
    ax.view_init(30, -45)
    ax.set_xlabel('Number of terms')
    ax.set_ylabel('x in exp(x)')
    ax.set_zlabel('exp(x)')
    plt.legend()
    fstr = 'data/exp_val_wire_1.png'
    plt.savefig(fstr)

    ax.view_init(30, -90)
    fstr = 'data/exp_val_wire_2.png'
    plt.savefig(fstr)

    # figure 4 ----------------------------------------------------------------
    fig = plt.figure()
    plt.plot(p, label='10^i')
    plt.plot(f, label='i!')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()

    plt.xlabel('i (log scale)')
    plt.ylabel('value (log scale)')
    fstr = 'data/f_p.png'
    plt.savefig(fstr)

    # plt.figure()
    # pno = 0
    # for p in range(pno, pno + 50):
    #     pstr = 'L = ' + str(p)
    #     print pstr
    #     plt.plot(inputs, data[p, :], label=pstr)
    #
    # plt.legend()
    # plt.plot([-10.2, 0], [0.0001, 0.0001])
    # plt.axis([-10.2, 0, 0, 0.001])

    plt.show()
    plt.close()
