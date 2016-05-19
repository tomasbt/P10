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

def myGIF(I,p,r,eps):
    '''
    This my implementation of a Guided image filter

    input:
        I : guidance image
        p : filter input
        r : window size / radius
        eps : epsilon. determines whether a or b have the most influence
    output:
        q : filter output

    q = a*I+b

    step 1: m_I = mean(I)
            m_p = mean(p)
            cor_II = mean(I*I)
            cor_Ip = mean(I*p)

    step 2: var_I = cor_II - m_I*m_I
            cov_Ip = cor_Ip - m_I*m_p

    step 3: a = cov_Ip/(var_I+eps)
            b = m_p - a*m_I

    step 4: m_a = mean(a)
            m_b = mean(b)

    step 5: q = m_a * I + m_b
    '''
    hei, wid = p.shape
    N = myBF(np.ones((hei,wid)),r)*1.0

    # step 1
    # m_I
    m_Ir = myBF(I[:,:,0],r) / N
    m_Ig = myBF(I[:,:,1],r) / N
    m_Ib = myBF(I[:,:,2],r) / N

    # m_p
    m_p = myBF(p,r)/N

    # cor_II
    # ??

    # cor_Ip
    m_Ipr = myBF(I[:,:,0]*p,r) / N
    m_Ipg = myBF(I[:,:,1]*p,r) / N
    m_Ipb = myBF(I[:,:,2]*p,r) / N

    # step 2
    # var_I
    var_Irr = myBF(I[:,:,0]*I[:,:,0],r) / N - m_Ir*m_Ir
    var_Irg = myBF(I[:,:,0]*I[:,:,1],r) / N - m_Ir*m_Ig
    var_Irb = myBF(I[:,:,0]*I[:,:,2],r) / N - m_Ir*m_Ib
    var_Igg = myBF(I[:,:,1]*I[:,:,1],r) / N - m_Ig*m_Ig
    var_Igb = myBF(I[:,:,1]*I[:,:,2],r) / N - m_Ig*m_Ib
    var_Ibb = myBF(I[:,:,2]*I[:,:,2],r) / N - m_Ib*m_Ib

    # cov_Ip
    cov_Ipr = m_Ipr - m_Ir * m_p
    cov_Ipg = m_Ipg - m_Ig * m_p
    cov_Ipb = m_Ipb - m_Ib * m_p

    # step 3
    # a
    a = np.zeros((hei,wid,3))
    for y in range(hei):
        for x in range(wid):
            sigma = np.asarray([[var_Irr[y,x],var_Irg[y,x],var_Irb[y,x]],
                                [var_Irg[y,x],var_Igg[y,x],var_Igb[y,x]],
                                [var_Irb[y,x],var_Igb[y,x],var_Ibb[y,x]]])
            # sigma = sigma + eps * np.identity(3)

            cov_Ip = [cov_Ipr[y,x], cov_Ipg[y,x], cov_Ipb[y,x]]
            a[y,x,:] = np.dot(cov_Ip,np.linalg.inv(sigma + eps * np.identity(3)))

    # b
    b = m_p - a[:,:,0]*m_Ir - a[:,:,1]*m_Ig - a[:,:,2]*m_Ib

    # step 4:
    # m_a
    m_ar = myBF(a[:,:,0],r)/N
    m_ag = myBF(a[:,:,1],r)/N
    m_ab = myBF(a[:,:,2],r)/N

    # m_b
    m_b = myBF(b,r)/N

    # step 5:
    # q
    q = m_ar*I[:,:,0]+m_ag*I[:,:,1]+m_ab*I[:,:,2]+m_b

    return q

def myBF(data,r):
    '''
    my implementation of a box filter

    WORKING AS INTENDED

    input:
        data : the data being filtered
        r : window size / radius
    output:
        out : output
    '''
    hei, wid = data.shape
    out = np.zeros(data.shape)

    # cummelative sum over y-axis
    imc = np.cumsum(data,0)

    # calculate difference over y-axis
    out[0:r+1,:] = imc[r:2*r+1,:]
    out[r+1:hei-r,:] = imc[2*r+1:hei,:] - imc[0:hei-2*r-1,:]
    out[hei-r:hei] = ml.repmat(imc[hei-1,:],r,1) - imc[hei-2*r-1:hei-r-1,:]

    # cummelative sum over x-axis
    imc = np.cumsum(out,1)

    # calculate differences over x-axis
    out[:,0:r+1] = imc[:,r:2*r+1]
    out[:,r+1:wid-r] = imc[:,2*r+1:wid] - imc[:,0:wid-2*r-1]
    out[:,wid-r:wid] = ml.repmat(imc[:,wid-1],1,r).reshape(hei,r) - imc[:,wid-2*r-1:wid-r-1]

    return out


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
             'ted' : ['data/usable/tedl.ppm','data/usable/tedr.ppm',59],
             'mot' : ['data/usable/motl.ppm','data/usable/motr.ppm',70],
             'tsu' : ['data/usable/tsul.ppm','data/usable/tsur.ppm',30],
             'ven' : ['data/usable/venl.ppm','data/usable/venr.ppm',32]}

    # set constants
    image = 'mot'
    al = 0.5

    maxDisp = fdict[image][2]
    r = 9
    eps = 0.0001*255
    lim = 2
    tB = 3.0
    tC = 7.0
    tG = 2.0
    fnamel = fdict[image][0]
    fnamer = fdict[image][1]
    # load images
    Il = readcolorppm(fnamel)#'data/tsukuba/tsconl.ppm')
    Ir = readcolorppm(fnamer)#'data/tsukuba/tsconr.ppm')
    Il = Il
    Ir = Ir
    # Il = Il*1.0
    # Ir = Ir*1.0
    Limg = cv2.imread(fnamel)#'data/tsukuba/tsconl.ppm')
    Rimg = cv2.imread(fnamer)#'data/tsukuba/tsconr.ppm')
    LimgG = cv2.cvtColor(Limg, cv2.COLOR_BGR2GRAY)
    RimgG = cv2.cvtColor(Rimg, cv2.COLOR_BGR2GRAY)
    Ilg = rgb2gray(Il)
    Irg = rgb2gray(Ir)

    # mirror images
    Il_1 = Il[:,::-1,:]
    Ir_1 = Ir[:,::-1,:]


    Rimg_1 = Rimg[:,::-1,:]
    Limg_1 = Limg[:,::-1,:]

    # compute gradient in X-direction
    fx_l = np.gradient(Ilg)[1]
    fx_r = np.gradient(Irg)[1]

    fx_l_1 = fx_l[:,::-1]
    fx_r_1 = fx_r[:,::-1]

    m, n, c = Il.shape

    dispVol = np.ones((m,n,maxDisp))*tB
    dispVol1 = np.ones((m,n,maxDisp))*tB

    print 'Starting cost calculation. Time taken so far', time.time()-start, 'seconds'

    for d in range(maxDisp):
        # calculate SAD
        tmp = np.ones((m,n,c))*tB
        tmp[:,d:n,:] = Ir[:,0:n-d,:]
        c_color = np.abs((tmp - Il)**2)
        c_color = np.sum(c_color,2)
        c_color = np.minimum(c_color,tC)

        # calculate gradient cost
        tmp = np.ones((m,n))*tB
        tmp[:,d:n] = fx_r[:,0:n-d]
        c_grad = np.abs(tmp - fx_l)
        c_grad = np.minimum(c_grad,tG)

        # calculate total cost
        c_tot = al*c_color + (1-al)*c_grad

        # do the same for the other view
        # SAD
        tmp1 = np.ones((m,n,c))*tB
        tmp1[:,d:n] = Il_1[:,0:n-d]
        c1_color = np.abs((tmp1 - Ir_1)**2)
        c1_color = np.sum(c1_color,2)
        c1_color = np.minimum(c1_color,tC)

        # Grad
        tmp1 = np.ones((m,n))*tB
        tmp1[:,d:n] = fx_l_1[:,0:n-d]
        c1_grad = np.abs(tmp1 - fx_r_1)
        c1_grad = np.minimum(c1_grad, tG)

        # total
        c1_tot = al*c1_color + (1-al)*c1_grad


        # set values
        dispVol[:,:,d] = c_tot
        dispVol1[:,:,d] = c1_tot[:,::-1]


    print 'Starting Guided image filter. Time taken so far', time.time()-start, 'seconds'


    Il_gf = cv2.ximgproc.createGuidedFilter(Limg,r,eps)
    Ir_gf = cv2.ximgproc.createGuidedFilter(Limg_1,r,eps)
    q = np.zeros((m,n),dtype=np.float32)
    q1 = np.zeros((m,n),dtype=np.float32)

    # guided image filter
    for d in range(maxDisp):
        p = dispVol[:,:,d].astype(np.float32)
        p1 = dispVol1[:,:,d].astype(np.float32)

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


    plt.figure()
    plt.imshow(labels_right)
    # left - right consistency check
    # Y = ml.repmat(np.arange(m).reshape(1,m), 1, n)
    # X = ml.repmat(np.arange(n), m, 1)
    # XX = X - labels_left
    # XX[XX<1] = 1
    # # indices = sub2indWrap([m,n],Y,X)

    final_labels = 1*labels_left

    # final_labels[np.abs(labels_left-labels_right[indices])>=1] = -1

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

    for y in range(m):
        for x in range(maxDisp,0,-1):
            if final_labels_filled[y][x] <=0:
                final_labels_filled[y][x] = final_labels_filled[y][x+1]


    plt.figure()
    plt.imshow(final_labels_filled,cmap=plt.cm.gray)

    plt.figure()
    plt.imshow(final_labels,cmap=plt.cm.gray)
    fstr = 'data/res/'+image+'_ssd'+str(al)+'.jpg'
    plt.imsave(fstr,final_labels,cmap=plt.cm.gray)

    if image == 'mot':
        fstr = 'data/res/'+image+'_ssd'+str(al)+'.pfm'
        file = open(fstr,'wb')
        save_pfm(file, final_labels_filled.astype('float32'), scale = 1)
        file.close()



    print 'Script ended. Time taken:',time.time()-start,'seconds'
