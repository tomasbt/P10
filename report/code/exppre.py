# test af exp precision

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
        e = e + (x**n) / (math.factorial(n))

    return e

if __name__ == '__main__' or True:
    L = 5
    sig = 25.0
    inputs = -np.arange(256) / sig  # 255.0 / sig
    print inputs
    for i in range(len(inputs)):
        print myExp(inputs[i], L), np.exp(inputs[i])
        print np.abs(myExp(inputs[i], L) - np.exp(inputs[i]))
        if np.abs(myExp(inputs[i], L) - np.exp(inputs[i])) > 0.1:
            print 'Error 1', i, inputs[i]
            break
        if myExp(inputs[i], L) < 0:
            print 'Error 2', i, inputs[i]
