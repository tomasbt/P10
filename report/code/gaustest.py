# gaustest.py

import numpy as np
import scipy as sp
import scipy.signal as sig


def my_pyrDown(img):
    '''
    My implementation of a vertical downscaling using image pyramids
    '''
    knl = sig.get_window(('gaussian', 1.0), 5).reshape(5, 1)
    res = sig.convolve2d(img, knl)
    res = res[2:-2:2, :]  # remove padding and remove even rows
    return res


def my_pyrUp(img):
    '''
    My implementation of a vertical upscaling using image pyramids
    '''
    res = np.zeros((img.shape[0] * 2, img.shape[1]))
    res[::2, :] = img
    knl = sig.get_window(('gaussian', 1.0), 5).reshape(5, 1)*2
    res = sig.convolve2d(res, knl)
    res = res[2:-2, :]  # remove padding
    return res

if __name__ == '__main__':
    print 'start'

    img = np.asarray([[1, 2, 1, 1, 1, 3, 4],
                      [2, 1, 1, 1, 0, 0, 0],
                      [2, 1, 2, 1, 0, 2, 0],
                      [2, 1, 2, 2, 2, 3, 0],
                      [2, 1, 4, 5, 6, 0, 0],
                      [2, 1, 2, 1, 0, 1, 0],
                      [2, 1, 1, 1, 0, 2, 0]])

    img_d = my_pyrDown(img)

    img_u = my_pyrUp(img_d)

    print img_u
