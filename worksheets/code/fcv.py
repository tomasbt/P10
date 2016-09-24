# fast cost volumne

import numpy as np
# import re
import time
# import sys
# sys.path.append('/Users/tt/.virtualenvs/cv/lib/python2.7/site-packages')
# import cv2
from matplotlib import pyplot as plt
import numpy.matlib as ml


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
    out[hei-r:hei] = np.matlib.repmat(imc[hei-1,:],r,1) - imc[hei-2*r-1:hei-r-1,:]

    # cummelative sum over x-axis
    imc = np.cumsum(out,1)

    # calculate differences over x-axis
    out[:,0:r+1] = imc[:,r:2*r+1]
    out[:,r+1:wid-r] = imc[:,2*r+1:wid] - imc[:,0:wid-2*r-1]
    out[:,wid-r:wid] = np.matlib.repmat(imc[:,wid-1],1,r).reshape(hei,r) - imc[:,wid-2*r-1:wid-r-1]

    return out

if __name__ == '__main__' or True:

    # imdata = np.asarray([[16, 2, 3, 13],[5, 11, 1, 8],[9, 7, 6, 12],[4, 14, 15, 1]])

    # imdata = np.asarray([[35,1,6,26,19,24],
    #                     [3,32,7,21,23,25],
    #                     [31,9,2,22,27,20],
    #                     [8,28,33,17,10,15],
    #                     [30,5,34,12,14,16],
    #                     [4,36,29,13,18,11]])

    start = time.time()
    al = 0.5
    maxDisp = 30
    r = 9
    eps = 0.0001
    lim = 2

    # Limg = cv2.imread('data/tsukuba/tsconl.ppm')
    # Rimg = cv2.imread('data/tsukuba/tsconr.ppm')
    # LimgG = cv2.cvtColor(Limg, cv2.COLOR_BGR2GRAY)
    # RimgG = cv2.cvtColor(Rimg, cv2.COLOR_BGR2GRAY)
    Limg = readcolorppm('data/tsukuba/tsconl.ppm')
    Rimg = readcolorppm('data/tsukuba/tsconr.ppm')
    LimgG = np.floor(rgb2gray(Limg))
    RimgG = np.floor(rgb2gray(Rimg))

    print time.time()-start

    Lgrad = np.gradient(LimgG)
    Rgrad = np.gradient(RimgG)

    # print time.time()-start

    # LimgGF = cv2.ximgproc.createGuidedFilter(Limg,r,eps)
    # RimgGF = cv2.ximgproc.createGuidedFilter(Rimg,r,eps)

    # print time.time()-start

    costMat = np.zeros(LimgG.shape+(maxDisp,),np.float32)
    costMat1 = np.zeros(LimgG.shape+(maxDisp,),np.float32)
    for y in range(Rimg.shape[0]):
        for x in range(Rimg.shape[1]):
            maxD = maxDisp if x+maxDisp <= Rimg.shape[1]-1 else Rimg.shape[1]-x-1
            for d in range(maxD):
                costMat[y][x][d] = al*np.sum(np.abs(Rimg[y][x]-Limg[y][x+d]))+ \
                        (1-al)*(np.abs(Rgrad[1][y][x]-Lgrad[1][y][x+d])+ \
                        np.abs(Rgrad[0][y][x]-Lgrad[0][y][x+d]))
            maxD = maxDisp if x-maxDisp >= 0 else x
            for d in range(maxD):
                costMat1[y][x][d] = al*np.sum(np.abs(Limg[y][x]-Rimg[y][x-d]))+ \
                    (1-al)*(np.abs(Lgrad[1][y][x]-Rgrad[1][y][x-d])+ \
                    np.abs(Lgrad[0][y][x]-Rgrad[0][y][x-d]))

    print time.time()-start
    dispvol = np.zeros((maxDisp,)+LimgG.shape,np.float32)
    dispvol1 = np.zeros((maxDisp,)+LimgG.shape,np.float32)
    # for d in range(maxDisp):
    #     RimgGF.filter(costMat[:,:,d],dispvol[d])
    #     LimgGF.filter(costMat1[:,:,d],dispvol1[d])

    print 'STARTING my GUIDED filter '

    for d in range(maxDisp):
        dispvol[d,:,:] = myGIF(Rimg,costMat[:,:,d],r,eps)
        dispvol1[d,:,:] = myGIF(Limg,costMat[:,:,d],r,eps)

    print time.time()-start
    dispmap = np.zeros(RimgG.shape)
    dispmap1 = np.zeros(LimgG.shape)
    dispmap2 = np.zeros(LimgG.shape)
    for y in range(Rimg.shape[0]):
        for x in range(Rimg.shape[1]):
            dispmap[y][x] = np.argmin(dispvol[:,y,x])
            dispmap1[y][x] = np.argmin(dispvol1[:,y,x])
            if dispmap1[y][x]-lim < dispmap[y][x] < dispmap1[y][x]+lim:
                dispmap2[y][x] = dispmap[y][x]
            else:
                dispmap2[y][x] = 0
    plt.figure()
    plt.imshow(dispmap)
    plt.figure()
    plt.imshow(dispmap1)
    plt.figure()
    plt.imshow(dispmap2)

    print 'It took', time.time()-start
