# scale moto image
import re
import sys
from matplotlib import pyplot as plt
import scipy
from skimage import io
import numpy as np


def load_pfm(file):
    '''
    Load a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.
    '''
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    return np.reshape(data, shape), scale

if __name__ == "main" or True:
    imgf = '../figures/mot_GT.jpg'

    f = open('motdisp0GT.pfm')
    img, s = load_pfm(f)
    f.close()

    img = img[::-1, :]
    ii = np.max(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] == ii:
                img[y, x] = 0
    img[0, 0] = 75
    plt.imshow(img, cmap=plt.cm.gray)
    plt.figure()
    fstr = '../figures/mot_gt.png'
    plt.imsave(fstr, img, cmap=plt.cm.gray)
    print "image saved as:" + fstr
    plt.show()
