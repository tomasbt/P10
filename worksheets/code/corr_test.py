# correlation test

import numpy as np
import time
from matplotlib import pyplot as plt
import sys
sys.path.append('/Users/tt/.virtualenvs/cv/lib/python2.7/site-packages')
import cv2


if __name__ == '__main__' or True:
    a = np.random.randint(0,256,(5,10))
    b = np.random.randint(0,256,(5,10))
    a = a.astype(np.uint8)
    b = b.astype(np.uint8)
    b[:,5:7] = a[:,0:2]-3

    res = cv2.matchTemplate(b,a[:,0:2],cv2.TM_CCORR_NORMED)

    for x in range(9):
        print a[0,0:2], b[0,0+x:2+x]
        print a[1,0:2], b[1,0+x:2+x]
        print a[2,0:2], b[2,0+x:2+x], res[:,x]
        print a[3,0:2], b[3,0+x:2+x]
        print a[4,0:2], b[4,0+x:2+x]
        print ''
