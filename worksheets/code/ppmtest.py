### read ppm test

import re
import numpy
import matplotlib.cm as cm

# def read_ppm(filename, byteorder='>'):
#     """Return image data from a raw PGM file as numpy array.
#
#     Format specification: http://netpbm.sourceforge.net/doc/pgm.html
#
#     """
#     with open(filename, 'rb') as f:
#         buffer = f.read()
#     try:
#         header, width, height, maxval = re.search(
#             b"(^P3\s(?:\s*#.*[\r\n])*"
#             b"(\d+)\s(?:\s*#.*[\r\n])*"
#             b"(\d+)\s(?:\s*#.*[\r\n])*"
#             b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
#     except AttributeError:
#         raise ValueError("Not a raw PGM file: '%s'" % filename)
#     print buffer
#     return numpy.frombuffer(buffer,
#                             dtype='u1' if int(maxval) < 256 else byteorder+'u2',
#                             count=int(width)*int(height)*3,
#                             offset=len(header)
#                             ).reshape((int(height), int(width), 3))
#
#

def readcolorppm(filename):
    '''
    Reads the specified file and returns an array containing image data
    can't handle comments in the file
    '''
    f = open(filename)
    color = f.readline().splitlines()
    size_x, size_y = f.readline().split()
    max = f.readline().splitlines()
    data = f.read().split()
    data = map(int,data)
    return numpy.asarray(data).reshape(int(size_y),int(size_x),3)

if __name__ == "__main__":
    from matplotlib import pyplot
    image = readcolorppm('data/tsukuba/tsconl.ppm')
    pyplot.imshow(image)
    pyplot.show()
