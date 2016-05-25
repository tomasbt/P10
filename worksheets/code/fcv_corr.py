# # fast fast cost volume simulation fcv.py
#

# fast fast cost volume simulation fcv.py

# imports
import numpy as np
import time
from matplotlib import pyplot as plt
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

def sub2ind(array_shape, rows, cols):
    '''
    Found on stackoverview
    '''

    return rows*array_shape[1] + cols

def sub2indWrap(array_shape,rows_arr,cols_arr):
    '''
    wrapper for sub2ind
    '''
    ind = np.zeros(array_shape)

    for y in range(array_shape[0]):
        for x in range(array_shape[1]):
            ind[y][x] = sub2ind(array_shape,y,x)

    return ind.astype(int)

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
             'conf' : ['data/usable/conlf.ppm','data/usable/conrf.ppm',59*4],
             'ted' : ['data/usable/tedl.ppm','data/usable/tedr.ppm',59],
             'tedf' : ['data/usable/tedlf.ppm','data/usable/tedrf.ppm',59*4],
             'mot' : ['data/usable/motl.ppm','data/usable/motr.ppm',70],
             'tsu' : ['data/usable/tsul.ppm','data/usable/tsur.ppm',30],
             'nku' : ['data/usable/nkul.ppm','data/usable/nkur.ppm',130],
             'ven' : ['data/usable/venl.ppm','data/usable/venr.ppm',32]}

    # set constants
    image = 'tsu'
    maxDisp = fdict[image][2] # gain maxDisp from the dictionary
    r = 36
    cr = 2
    eps = 0.0001
    lim = 5
    tB = 3.0/255
    tC = 7.0/255
    tG = 2.0/255
    g_c = 0.1
    g_d = 9
    r_median = 19

    fnamel = fdict[image][0]
    fnamer = fdict[image][1]

    # load images
    Limg = cv2.imread(fnamel)
    Rimg = cv2.imread(fnamer)

    # mirror images
    Limg_1 = Limg[:,::-1,:]
    Rimg_1 = Rimg[:,::-1,:]

    # get size of the images
    m, n, c = Limg.shape

    print 'Starting cost calculation. Time taken so far', time.time()-start, 'seconds'

    # preform normalized cross correlation
    c_color = np.ones((m,n,maxDisp))*(1-tB)
    c1_color = np.ones((m,n,maxDisp))*(1-tB)
    for y in range(cr,m-cr):
        for x in range(cr,n-cr):
            d = maxDisp-1 if x >= cr+maxDisp else x-cr
            c_color[y,x,maxDisp-d-1:] = cv2.matchTemplate(Rimg[y-cr:y+cr+1,x-cr-d:x+cr+1], Limg[y-cr:y+cr+1,x-cr:x+cr+1], cv2.TM_CCORR_NORMED)[0]
            c1_color[y,x,maxDisp-d-1:] = cv2.matchTemplate(Limg_1[y-cr:y+cr+1,x-cr-d:x+cr+1], Rimg_1[y-cr:y+cr+1,x-cr:x+cr+1], cv2.TM_CCORR_NORMED)[0]

    c_color = 1-c_color[:,:,::-1]
    c1_color = 1-c1_color[:,::-1,::-1]

    print 'time taken', time.time()-start

    # Minimization before guided image filtering
    labels_left1 = np.argmin(c_color,axis=2)
    labels_right1 = np.argmin(c1_color,axis=2)

    # plot figures before guided image filtering
    plt.figure()
    plt.title('Left labels before GIF')
    plt.imshow(labels_left1,cmap=plt.cm.gray)
    plt.figure()
    plt.title('Right labels before GIF')
    plt.imshow(labels_right1,cmap=plt.cm.gray)
    final_labels1 = 1*labels_left1

    # find occluded points
    for y in range(m):
        for x in range(n):
            if np.abs(labels_left1[y][x]-labels_right1[y][x-labels_left1[y][x]])>=lim:
                final_labels1[y][x] = -1
    plt.figure()
    plt.title('Final labels before GIF')
    plt.imshow(final_labels1,cmap=plt.cm.gray)
    fstr = 'data/res/'+image+'_fcv_corr_noGIF_cr'+str(cr)+'_r'+str(r)+'.jpg'
    plt.imsave(fstr,final_labels,cmap=plt.cm.gray)

    print 'Starting Guided image filter. Time taken so far', time.time()-start, 'seconds'

    Il_gf = cv2.ximgproc.createGuidedFilter(Limg,r,eps)
    Ir_gf = cv2.ximgproc.createGuidedFilter(Rimg_1,r,eps)
    q = np.zeros((m,n),dtype=np.float32)
    q1 = np.zeros((m,n),dtype=np.float32)
    dispVol = np.ones((m,n,maxDisp))*(1-tB)
    dispVol1 = np.ones((m,n,maxDisp))*(1-tB)

    # guided image filter
    for d in range(maxDisp):
        p = c_color[:,:,d].astype(np.float32)
        p1 = c1_color[:,:,d].astype(np.float32)

        # q = myGIF(Il,p,r,eps)
        Il_gf.filter(p,q)

        p1 = p1[:,::-1]

        # q1 = myGIF(Il_1, p1, r, eps)
        Ir_gf.filter(p1,q1)

        dispVol[:,:,d] = q
        dispVol1[:,:,d] = q1[:,::-1]


    print 'Starting minimization. Time taken so far', time.time()-start, 'seconds'

    # minimization
    labels_left = np.argmin(dispVol,axis=2)
    labels_right = np.argmin(dispVol1,axis=2)

    final_labels = 1*labels_left

    # find occluded labels
    for y in range(m):
        for x in range(n):
            if np.abs(labels_left[y][x]-labels_right[y][x-labels_left[y][x]])>=lim:
                final_labels[y][x] = -1

    # fill occluded points - nda
    final_labels_filled = 1*final_labels
    for y in range(m):
        for x in range(n):
            if final_labels_filled[y][x] <= 0:
                final_labels_filled[y][x] = final_labels_filled[y][x-1]
    # fill border occluded points
    for y in range(m):
        for x in range(maxDisp,0,-1):
            if final_labels_filled[y][x] <=0:
                final_labels_filled[y][x] = final_labels_filled[y][x+1]

    # Print figures and save figures
    plt.figure()
    plt.title('Final labels filled after GIF')
    plt.imshow(final_labels_filled,cmap=plt.cm.gray)

    plt.figure()
    plt.title('Final labels after GIF')
    plt.imshow(final_labels,cmap=plt.cm.gray)
    fstr = 'data/res/'+image+'_fcv_corr_cr'+str(cr)+'_r'+str(r)+'.jpg'
    plt.imsave(fstr,final_labels,cmap=plt.cm.gray)

    if image == 'mot':
        fstr = 'data/res/'+image+'_fcv_corr_cr'+str(cr)+'_r'+str(r)+'.pfm'
        file = open(fstr,'wb')
        save_pfm(file, final_labels_filled.astype('float32'), scale = 1)
        file.close()



    print 'Script ended. Time taken:',time.time()-start,'seconds'
