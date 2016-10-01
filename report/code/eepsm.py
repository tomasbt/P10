import numpy as np
import time
from matplotlib import pyplot as plt
import sys

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


# Calculate hamming distance ---------------------------------------------------
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
    This function is a implementation of the census transform.

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
        return np.exp(-np.abs(img[y][x]-img[y][x+1])/sigma)

    elif direct == 'l':
        # If the direction is left then:
        return np.exp(-np.abs(img[y][x]-img[y][x-1])/sigma)

    elif direct == 't':
        # If the direction is top then:
        return np.exp(-np.abs(img[y][x]-img[y+1][x])/sigma)

    elif direct == 'b':
        # If the direction is bottom then:
        return np.exp(-np.abs(img[y][x]-img[y-1][x])/sigma)
    else:
        # Error
        return None


def save_pfm(file, image, scale = 1):
    '''
    Saves image to pfm file.

    Found at https://gist.github.com/chpatrick/8935738
    '''

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

# main loop --------------------------------------------------------------------################
if __name__ == "main" or True:

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

    lim = 2
    al = 0.1
    maxDisp = fdict[image][2]
    consistancy_check = False

    fnamel = fdict[image][0]
    fnamer = fdict[image][1]

    # load images and normalise the data
    Limg = readcolorppm(fnamel)
    Rimg = readcolorppm(fnamer)
    LimgG = np.floor(rgb2gray(Limg))
    RimgG = np.floor(rgb2gray(Rimg))

    ## Lav 4 matricer som indeholder Aggregation for de 4 retninger
    aggre_L = np.zeros(LimgG.shape+(maxDisp,))  # Going from right to left
    aggre_R = np.zeros(LimgG.shape+(maxDisp,))  # Going from left to right
    aggre_T = np.zeros(LimgG.shape+(maxDisp,))  # Going from bottom to top
    aggre_B = np.zeros(LimgG.shape+(maxDisp,))  # Going from top to bottom

    # calculate cost for each pixel
    costMat = np.zeros(RimgG.shape+(maxDisp,))
    for y in range(1,Limg.shape[0]-1):
        for x in range(1,Limg.shape[1]-1):
            maxD = maxDisp if x-maxDisp >= 0 else x
            for d in range(maxD):
                costMat[y][x][d] = al*np.sum(np.abs(Limg[y][x]-Rimg[y][x-d]))+\
                        (1-al)*Hamm(ctrans(LimgG[y-1:y+2,x-1:x+2]),
                        ctrans(RimgG[y-1:y+2,x-1-d:x+2-d]))


    # Calculate left scan values
    for y in range(Limg.shape[0]):
        for x in range(Limg.shape[1],0,-1):
            maxD = maxDisp if x-maxDisp >= 0 else x
            for d in range(maxD):
                if x == 0:
                    continue
                elif x >= Limg.shape[1]-1:
                    continue
                if y == 0:
                    continue
                elif y == Limg.shape[0]-1:
                    continue

                aggre_L[y][x][d] = costMat[y][x][d] + permeability(LimgG,x,y,direct='l')*aggre_L[y][x+1][d]


    # horizontal aggregation
    horz_aggre = np.zeros(LimgG.shape+(maxDisp,))
    for y in range(1,Limg.shape[0]-1):
        for x in range(0,Limg.shape[1]-2):
            maxD = maxDisp if x-maxDisp >= 0 else x
            for d in range(maxD):
                if x == 0:
                    aggre_R[x][y][d] = costMat[y][x][d]
                    continue
                    horz_aggre[y][x][d] = costMat[y][x][d] + \
                            permeability(LimgG,x+1,y,direct='l')*aggre_L[y][x+1][d]
                    continue
                elif x >= Limg.shape[1]-2:
                    continue
                if y == 0:
                    continue
                elif y >= Limg.shape[0]-2:
                    continue
                aggre_R[y][x][d] = costMat[y][x][d] + permeability(LimgG,x,y,direct='r')*aggre_R[y][x-1][d]

                horz_aggre[y][x][d] = costMat[y][x][d] + \
                        permeability(LimgG,x,y,direct='l')*aggre_R[y][x-1][d] + \
                        permeability(LimgG,x,y,direct='r')*aggre_L[y][x+1][d]

    # calculate top scan values - bottom to top
    for y in range(Limg.shape[0]-1,0,-1):
        for x in range(Limg.shape[1]):
            maxD = maxDisp if x-maxDisp >= 0 else x
            for d in range(maxD):
                if x == 0:
                    continue
                elif x >= Limg.shape[1]-1:
                    continue
                if y == 0:
                    continue
                elif y >= Limg.shape[0]-1:
                    aggre_T[y][x][d] = horz_aggre[y][x][d]
                    continue

                aggre_T[y][x][d] = horz_aggre[y][x][d] + permeability(LimgG,x,y,direct='t')*aggre_T[y+1][x][d]


    # combine vertical aggregation data
    total_aggre = np.zeros(LimgG.shape+(maxDisp,))
    for y in range(0,Limg.shape[0]):
        for x in range(0,Limg.shape[1]):
            maxD = maxDisp if x-maxDisp >= 0 else x
            for d in range(maxD):
                if x == 0:
                    continue
                elif x >= Limg.shape[1]-1:
                    continue
                if y == 0:
                    aggre_B[y][x][d] = horz_aggre[y][x][d]
                    continue
                elif y >= Limg.shape[0]-1:
                    continue

                aggre_B[y][x][d] = horz_aggre[y][x][d] + permeability(LimgG,x,y,direct='t')*aggre_B[y-1][x][d]
                total_aggre[y][x][d] = horz_aggre[y][x][d] + \
                        permeability(LimgG,x,y,direct='b')*aggre_T[y+1][x][d] + \
                        permeability(LimgG,x,y,direct='t')*aggre_B[y-1][x][d]

    # generate disparity map
    dispmap = np.zeros(RimgG.shape)
    for y in range(Limg.shape[0]):
        for x in range(Limg.shape[1]):
            dispmap[y][x] = np.argmin(total_aggre[y][x])

    if consistancy_check:
        # calculate cost for each pixel
        costMat = np.zeros(RimgG.shape+(maxDisp,))
        for y in range(1,Rimg.shape[0]-1):
            for x in range(1,Rimg.shape[1]-1):
                # maxD = maxDisp if x-maxDisp >= 0 else x
                maxD = maxDisp if x+maxDisp <= Rimg.shape[1]-1 else Rimg.shape[1]-x-1
                for d in range(maxD):
                    costMat[y][x][d] = al*np.sum(np.abs(Rimg[y][x]-Limg[y][x+d]))+\
                            (1-al)*Hamm(ctrans(LimgG[y-1:y+2,x-1:x+2]),
                            ctrans(RimgG[y-1:y+2,x-1+d:x+2+d]))


        # Calculate left scan values
        for y in range(Rimg.shape[0]):
            for x in range(Rimg.shape[1],0,-1):
                # maxD = maxDisp if x-maxDisp >= 0 else x
                maxD = maxDisp if x+maxDisp <= Limg.shape[1]-1 else Limg.shape[1]-x-1
                for d in range(maxD):
                    if x == 0:
                        continue
                    elif x >= Rimg.shape[1]-1:
                        continue
                    if y == 0:
                        continue
                    elif y == Rimg.shape[0]-1:
                        continue

                    aggre_L[y][x][d] = costMat[y][x][d] + permeability(RimgG,x,y,direct='l')*aggre_L[y][x+1][d]


        # horizontal aggregation
        horz_aggre = np.zeros(RimgG.shape+(maxDisp,))
        for y in range(1,Rimg.shape[0]-1):
            for x in range(0,Rimg.shape[1]-2):
                # maxD = maxDisp if x-maxDisp >= 0 else x
                maxD = maxDisp if x+maxDisp <= Limg.shape[1]-1 else Limg.shape[1]-x-1
                for d in range(maxD):
                    if x == 0:
                        aggre_R[x][y][d] = costMat[y][x][d]
                        continue
                        horz_aggre[y][x][d] = costMat[y][x][d] + \
                                permeability(RimgG,x+1,y,direct='l')*aggre_L[y][x+1][d]
                        continue
                    elif x >= Rimg.shape[1]-2:
                        continue
                    if y == 0:
                        continue
                    elif y >= Rimg.shape[0]-2:
                        continue
                    aggre_R[y][x][d] = costMat[y][x][d] + permeability(RimgG,x,y,direct='r')*aggre_R[y][x-1][d]

                    horz_aggre[y][x][d] = costMat[y][x][d] + \
                            permeability(RimgG,x,y,direct='l')*aggre_R[y][x-1][d] + \
                            permeability(RimgG,x,y,direct='r')*aggre_L[y][x+1][d]

        # calculate top scan values - bottom to top
        for y in range(Rimg.shape[0]-1,0,-1):
            for x in range(Rimg.shape[1]):
                maxD = maxDisp if x-maxDisp >= 0 else x
                for d in range(maxD):
                    if x == 0:
                        continue
                    elif x >= Rimg.shape[1]-1:
                        continue
                    if y == 0:
                        continue
                    elif y >= Rimg.shape[0]-1:
                        aggre_T[y][x][d] = horz_aggre[y][x][d]
                        continue

                    aggre_T[y][x][d] = horz_aggre[y][x][d] + permeability(RimgG,x,y,direct='t')*aggre_T[y+1][x][d]


        # combine vertical aggregation data
        total_aggre = np.zeros(RimgG.shape+(maxDisp,))
        for y in range(0,Rimg.shape[0]):
            for x in range(0,Rimg.shape[1]):
                maxD = maxDisp if x-maxDisp >= 0 else x
                for d in range(maxD):
                    if x == 0:
                        continue
                    elif x >= Rimg.shape[1]-1:
                        continue
                    if y == 0:
                        aggre_B[y][x][d] = horz_aggre[y][x][d]
                        continue
                    elif y >= Rimg.shape[0]-1:
                        continue

                    aggre_B[y][x][d] = horz_aggre[y][x][d] + permeability(RimgG,x,y,direct='t')*aggre_B[y-1][x][d]
                    total_aggre[y][x][d] = horz_aggre[y][x][d] + \
                            permeability(RimgG,x,y,direct='b')*aggre_T[y+1][x][d] + \
                            permeability(RimgG,x,y,direct='t')*aggre_B[y-1][x][d]
        # generate disparity map
        dispmap2 = np.zeros(RimgG.shape)
        for y in range(Rimg.shape[0]):
            for x in range(Rimg.shape[1]):
                dispmap2[y][x] = np.argmin(total_aggre[y][x])

    # find and fill occlusions - nda
    dispmap_final = 1*dispmap
    m, n = dispmap.shape
    for y in range(m):
        for x in range(n):
            if np.abs(dispmap[y][x]-dispmap2[y][x-dispmap[y][x]])>=lim:
                dispmap_final[y][x] = -1

    # fill occluded points - nda
    dispmap_final_filled = 1*dispmap_final
    for y in range(m):
        for x in range(n):
            if dispmap_final_filled[y][x] <= 0:
                dispmap_final_filled[y][x] = dispmap_final_filled[y][x-1]
    for y in range(m): # bord
        for x in range(maxDisp,0,-1):
            if dispmap_final_filled[y][x] <=0:
                dispmap_final_filled[y][x] = dispmap_final_filled[y][x+1]

    print 'It took', time.time()-start
    # save the out as .png for the report
    plt.figure()
    fstr = 'data/res/'+image+'_eepsm_1.png'
    plt.imsave(fstr,dispmap,cmap=plt.cm.gray)
    print "image saved as:" + fstr
    plt.close()

    # save the out as .png for the report
    plt.figure()
    fstr = 'data/res/'+image+'_eepsm.png'
    plt.imsave(fstr,dispmap_final_filled,cmap=plt.cm.gray)
    print "image saved as:" + fstr
    
    plt.close()

    plt.figure()
    plt.imshow(dispmap,cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(dispmap2,cmap=plt.cm.gray)
    # save the ouput as .pfm if pfm files exist for the images
    if image == 'mot':
        fstr = 'data/res/'+image+'_eepsm.pfm'
        file = open(fstr,'wb')
        save_pfm(file, dispmap_final_filled.astype('float32'), scale = 1)
        file.close()
