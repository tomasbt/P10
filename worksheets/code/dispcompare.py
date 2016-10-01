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

if __name__ == 'main' or True:

    f = open('data/mot_GT.pfm')
    img, s = load_pfm(f)
    f.close()

    f = open('data/res/fcv_norm/mot_fcv_r9_al0.11.pfm')
    img2, s2 = load_pfm(f)
    f.close()

    img = np.flipud(img)

    if img.shape == img2.shape:
        print 'same shape'
    else:
        print 'SHAPE NOT EQUAL'


    Diff = 0
    numbofwrongpixels = 0
    occImg = np.zeros(img2.shape)
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
