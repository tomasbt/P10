# simple test of eespm

import numpy as np
import time

def Hamm(arr1,arr2):
    '''
    My own hamming distance for 8 bits / parts
    '''
    hammDst = 0
    for i in range(8):
        if arr1[i] != arr2[i]:
            hammDst = hammDst + 1

    return hammDst

def ctrans(data):
    '''
    This function is a reimplementation of the census transform.

    input 3x3 --> output 8x1
    '''

    censustransform = np.zeros(8)

    count = 0
    for y in range(3):
        for x in range(3):

            if y == 1 & x == 1:
                continue

            if data[y][x] < data[1][1]:
                censustransform[count] = 1
                count += 1

    return censustransform

def permeability(img,x,y,sigma=0.5,direct='r'):
    '''
    this function calculates permeability weight in the specified direction
    '''
    if direct == 'r':
        # If the direction is right then:
        return np.exp(-np.abs(int(img[y][x])-int(img[y][x+1]))/sigma)

    elif direct == 'l':
        # If the direction is right then:
        return np.exp(-np.abs(int(img[y][x])-int(img[y][x-1]))/sigma)

    elif direct == 't':
        # If the direction is right then:
        return np.exp(-np.abs(int(img[y][x])-int(img[y+1][x]))/sigma)

    elif direct == 'b':
        # If the direction is right then:
        return np.exp(-np.abs(int(img[y][x])-int(img[y-1][x+1]))/sigma)
    else:
        # Error
        return None

def sws_update(limg,rimg,x,y,f_data,alpha,d,direct='r'):
    '''
    Update rule for successive weighted sum

    limg   - data from the left image
    rimg   - data from the right image
    x, y   - coordinates for the current pixel
    direct - which scan order is going to be preformed. either 'r','l','t' or 'b'
    f_data - former data
    alpha  - parameter for cost calculation
    d      - Disparity
    '''

    c_sad = np.abs(rimg[y][x]-limg[y][x+d])
    c_d = alpha*c_sad+(1-alpha)*Hamm(ctrans(rimg[y-1:y+2,x-1:x+2]),
                                    ctrans(limg[y-1:y+2,x-1+d:x+2+d]))

    if direct == 'r':
        # if the direction is right then do the following

        curr_c = c_d + permeability(rimg,x,y,direct='r')*f_data

    elif direct == 'l':
        # if the direction is left then do the following

        curr_c = c_d + permeability(rimg,x,y,direct='l')*f_data

    elif direct == 't':
        # if the direction is left then do the following

        curr_c = c_d + permeability(rimg,x,y,direct='t')*f_data

    elif direct == 'b':
        # if the direction is left then do the following

        curr_c = c_d + permeability(rimg,x,y,direct='b')*f_data

    # print 'x,y:',x,y,'and d:',d,'and cost:',curr_c
    return curr_c

if __name__ == "main" or True:

    start = time.time()

    imgSz = 300
    maxval = 30
    CDisp = 3
    al = 0.5
    maxDisp = 30

    Limg = np.ceil(maxval*np.random.rand(imgSz,imgSz))

    Rimg = np.append(Limg[:,CDisp:],np.ceil(maxval*np.random.rand(imgSz,CDisp)),axis=1)

    # print "Left image is\n", Limg ,"\n"
    # print "Right image is\n", Rimg ,"\n"

    # C is the cost C[1][3] = C_1(3)

    ## Lav 4 matricer som indeholder Aggregation for de 4 retninger
    aggre_L = np.zeros(Limg.shape+(maxDisp,))  # Going from right to left
    aggre_R = np.zeros(Limg.shape+(maxDisp,))  # Going from left to right
    aggre_T = np.zeros(Limg.shape+(maxDisp,))  # Going from bottom to top
    aggre_B = np.zeros(Limg.shape+(maxDisp,))  # Going from top to bottom

    # Start by generating right scan values
    for y in range(Rimg.shape[0]):
        for x in range(Rimg.shape[1]):
            maxD = maxDisp if x+maxDisp <= Rimg.shape[1]-1 else Rimg.shape[1]-x-1
            for d in range(maxD):
                if x == 0:
                    continue
                elif x >= Rimg.shape[1]-1:
                    continue
                if y == 0:
                    continue
                elif y == Rimg.shape[0]-1:
                    continue

                # print 'dir = R','x,y:',x,y,'and d:',d
                aggre_R[y][x][d] = sws_update(Limg,Rimg,x,y,aggre_R[y][x-1][d],al,d,direct='r')

    # Calculate left scan values
    for y in range(Rimg.shape[0]):
        for x in range(Rimg.shape[1],0,-1):
            maxD = maxDisp if x+maxDisp <= Rimg.shape[1]-1 else Rimg.shape[1]-x-1
            for d in range(maxD):
                if x == 0:
                    continue
                elif x >= Rimg.shape[1]-1:
                    continue
                if y == 0:
                    continue
                elif y == Rimg.shape[0]-1:
                    continue


                # sws_update(limg,rimg,x,y,direct='r',f_data,alpha,d)
                # print 'dir = L','x,y:',x,y,'and d:',d
                aggre_L[y][x][d] = sws_update(Limg,Rimg,x,y,aggre_L[y][x+1][d],al,d,direct='l')

    # calculate top scan values
    for y in range(Rimg.shape[0],0,-1):
        for x in range(Rimg.shape[1]):
            maxD = maxDisp if x+maxDisp <= Rimg.shape[1]-1 else Rimg.shape[1]-x-1
            for d in range(maxD):
                if x == 0:
                    continue
                elif x >= Rimg.shape[1]-1:
                    continue
                if y == 0:
                    continue
                elif y >= Rimg.shape[0]-1:
                    continue

                # print 'dir = T','x,y:',x,y,'and d:',d
                aggre_T[y][x][d] = sws_update(Limg,Rimg,x,y,aggre_T[y+1][x][d],al,d,direct='t')

    # calculate bottom scan values
    for y in range(Rimg.shape[0]):
        for x in range(Rimg.shape[1]):
            maxD = maxDisp if x+maxDisp <= Rimg.shape[1]-1 else Rimg.shape[1]-x-1
            for d in range(maxD):
                if x == 0:
                    continue
                elif x >= Rimg.shape[1]-1:
                    continue
                if y == 0:
                    continue
                elif y == Rimg.shape[0]-1:
                    continue

                # print 'dir = B','x,y:',x,y,'and d:',d
                aggre_B[y][x][d] = sws_update(Limg,Rimg,x,y,aggre_B[y-1][x][d],al,d,direct='b')

    # combine values
    horz_aggre = np.zeros(Limg.shape+(maxDisp,))
    for y in range(1,Rimg.shape[0]-1):
        for x in range(1,Rimg.shape[1]-1):
            maxD = maxDisp if x+maxDisp <= Rimg.shape[1]-1 else Rimg.shape[1]-x-1
            for d in range(maxD):
                # horizontal data
                c_sad = np.abs(Rimg[y][x]-Limg[y][x+d])
                c_d = al*c_sad+(1-al)*Hamm(ctrans(Rimg[y-1:y+2,x-1:x+2]),
                                            ctrans(Limg[y-1:y+2,x-1+d:x+2+d]))
                horz_aggre[y][x][d] = c_sad + \
                        permeability(Rimg,x-1,y,direct='r')*aggre_R[y][x-1][d] + \
                        permeability(Rimg,x+1,y,direct='l')*aggre_L[y][x+1][d]


    print 'time taken:', time.time()-start,'secs'












# DARGH
