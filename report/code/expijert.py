# ijert exp implementation

import numpy as np


def negexp(Input):
    '''
    implementation of exp function from ijert article
    '''
    ip = Input * 1
    K = [5.5452, 2.7726, 1.3863, 0.6931, 0.2877, 0.1335,
         0.0645, 0.0317, 0.0157, 0.0078, 0.0039]
    exp = [256, 16, 4, 2, 3 / 2, 5 / 4, 9 / 8, 17 / 16,
           33 / 32, 65 / 64, 129 / 128]
    OP = 1
    for J in range(0, 11):
        Temp = ip - exp[J]
        if Temp < 0:
            OP = OP * K[J]
            ip = Temp
        print J, Temp, OP
    return OP

if __name__ == '__main__':
    val = 255
    print negexp(val), np.exp(val)

    K = [5.5452, 2.7726, 1.3863, 0.6931, 0.2877, 0.1335,
         0.0645, 0.0317, 0.0157, 0.0078, 0.0039]
    print np.exp(K)
