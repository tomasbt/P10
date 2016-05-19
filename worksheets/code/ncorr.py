# fast fast cost volume simulation fcv.py

# imports
import numpy as np
import time
from matplotlib import pyplot as plt
from numpy import matlib as ml
import sys
sys.path.append('/Users/tt/.virtualenvs/cv/lib/python2.7/site-packages')
import cv2


# functions:
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
    return np.asarray(data).reshape(int(size_y),int(size_x),3)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def save_pfm(file, image, scale = 1):
  color = None

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

# main function:
if __name__ == '__main__' or True:
    # start timer
    start = time.time()


    # filenames
    fdict = {'con' : ['data/usable/conl.ppm','data/usable/conr.ppm',59],
             'ted' : ['data/usable/tedl.ppm','data/usable/tedr.ppm',59],
             'mot' : ['data/usable/motl.ppm','data/usable/motr.ppm',70],
             'tsu' : ['data/usable/tsul.ppm','data/usable/tsur.ppm',30],
             'ven' : ['data/usable/venl.ppm','data/usable/venr.ppm',32]}

    # set constants
    image = 'mot'

    maxDisp = fdict[image][2]
    r = 2 # => 2*r+1 windows size
    lim = 2

    fnamel = fdict[image][0]
    fnamer = fdict[image][1]

    Limg = cv2.imread(fnamel)
    Rimg = cv2.imread(fnamer)

    # mirror images
    Rimg_1 = Rimg[:,::-1,:]
    Limg_1 = Limg[:,::-1,:]

    m, n, c = Limg.shape
    tB=3

    print 'Starting cost calculation. Time taken so far', time.time()-start, 'seconds'

    c_color = np.zeros((m,n,maxDisp))#*(tB/255.0)
    c1_color = np.zeros((m,n,maxDisp))#*(tB/255.0)
    for y in range(r*2,m-r):
        for x in range(r+2,n-r):
            d = maxDisp-1 if x >= r+maxDisp else x-r
            # for d in range(maxD):  #(image, templ, method[, result])
            c_color[y,x,0:d+1] = cv2.matchTemplate(Rimg[y-r:y+r+1,x-r-d:x+r+1], Limg[y-r:y+r+1,x-r:x+r+1], cv2.TM_CCORR_NORMED)[0]
            c1_color[y,x,0:d+1] = cv2.matchTemplate(Limg_1[y-r:y+r+1,x-r-d:x+r+1], Rimg_1[y-r:y+r+1,x-r:x+r+1], cv2.TM_CCORR_NORMED)[0]

    c_color = c_color[:,:,::-1]
    c1_color = c1_color[:,::-1,::-1]


    # minimization

    labels_left = np.argmax(c_color,axis=2)
    labels_right = np.argmax(c1_color,axis=2)
    labels_right_shifted = np.zeros((m,n))
    labels_right_shifted[:,0:n-maxDisp] = labels_right[:,maxDisp:n]

    plt.figure()
    plt.imshow(labels_left,cmap=plt.cm.gray)

    plt.figure()
    plt.imshow(labels_right,cmap=plt.cm.gray)


    # left - right consistency check
    # Y = ml.repmat(np.arange(m).reshape(1,m), 1, n)
    # X = ml.repmat(np.arange(n), m, 1)
    # XX = X - labels_left
    # XX[XX<1] = 1
    # # indices = sub2indWrap([m,n],Y,X)

    final_labels = 1*labels_left

    # final_labels[np.abs(labels_left-labels_right[indices])>=1] = -1

    for y in range(m):
        for x in range(n-maxDisp):
            if np.abs(labels_left[y][x]-labels_right[y][x-labels_left[y][x]])>=lim:
                final_labels[y][x] = -1

    # fill occluded points - nda
    final_labels_filled = 1*final_labels
    for y in range(m):
        for x in range(n):
            if final_labels_filled[y][x] <= 0:
                final_labels_filled[y][x] = final_labels_filled[y][x-1]

    for y in range(m):
        for x in range(maxDisp,0,-1):
            if final_labels_filled[y][x] <=0:
                final_labels_filled[y][x] = final_labels_filled[y][x+1]


    plt.figure()
    plt.imshow(final_labels_filled,cmap=plt.cm.gray)

    plt.figure()
    plt.imshow(final_labels,cmap=plt.cm.gray)
    fstr = 'data/res/'+image+'_ncorr.jpg'
    plt.imsave(fstr,final_labels,cmap=plt.cm.gray)
    print 'file saved as', fstr

    if image == 'mot':
        fstr = 'data/res/'+image+'_ncorr.pfm'
        file = open(fstr,'wb')
        save_pfm(file, final_labels_filled.astype('float32'), scale = 1)
        file.close()
        print 'file saved as', fstr



    print 'Script ended. Time taken:',time.time()-start,'seconds'
