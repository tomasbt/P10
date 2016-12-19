# exp pre calculate
import numpy as np
import math


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

if __name__ == '__main__' or True:

    L = 7
    sig = 38.1
    inputs = -np.arange(256) / sig  # 255.0 / sig
    pre = 0.001
    # print inputs

    # make a 3d plot

    # print L.shape, data.shape, inputs.shape
    maxVal = 1
    maxVal2 = 1  # 1.01
    # for x in range(len(inputs)):
    #     if np.abs(myExp(inputs[x], L) - np.exp(inputs[x])) > pre:
    #         print x, inputs[x], L, np.abs(myExp(inputs[x], L) -
    #                                       np.exp(inputs[x]))
    #         break

    x = inputs[-1]
    for l in range(1, 51):
        if np.abs(myExp(x, l) - np.exp(x)) < pre:
            print x, l, np.abs(myExp(x, l) - np.exp(x))
            break
        print l, myExp(x, l), np.exp(x), np.abs(myExp(x, l) - np.exp(x))

    x = (6.69)**23
    print x
    print np.ceil(math.log(x, 2))

    x = np.exp(inputs[-1])
    for i in range(1, 1000):
        if x > (2**(-i)):
            print i, inputs[-1], x, 2**(-i)
            break

        print '.', 1.0/x, 2**(-i)
