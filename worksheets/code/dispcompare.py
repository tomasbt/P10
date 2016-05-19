import numpy as np
import re
import sys
import matplotlib.pyplot as plt
'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def load_pfm(file):
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
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)

  return np.reshape(data, shape), scale

'''
Save a Numpy array to a PFM file.
'''
def save_pfm(filename, image, scale = 1):
  color = None

  file = open(filename,'w')

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)

  file.close()

  return

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    Gain from STACK OVERFLOW
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='int', #'u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


if __name__ == 'main' or True:

    f = open('disp0GT.pfm')
    img, s = load_pfm(f)
    f.close()

    f = open('data/res/mot.pfm')
    img2, s2 = load_pfm(f)
    f.close()
    
    img = np.flipud(img)

    # img2 = read_pgm('testf.pgm')


    if img.shape == img2.shape:
        print 'same shape'
    else:
        print 'SHAPE NOT EQUAL'


    Diff = 0
    numbofwrongpixels = 0
    # img = img[::-1,:]
    img[np.isinf(img)]=0
    lim = 5
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if np.abs(img2[y][x] - img[y][x]) > lim:
                Diff = Diff + np.abs(img2[y][x] - img[y][x])
                numbofwrongpixels += 1

    print Diff
    print numbofwrongpixels
    print img.size
    print 'percent error:', 1.0*numbofwrongpixels/img.size*100

    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(img2)


    # img[np.isinf(img)]=0
    # print np.max(img)
