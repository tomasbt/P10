# implementation of Efficient Edge-Preserving Stereo Matching in python
# grayscale version

import numpy as np
import re
from matplotlib import pyplot as plt
import time

# Read PGM ---------------------------------------------------------------------
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
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

# SAD --------------------------------------------------------------------------
def SADcost(limg,rimg,x,y,d):
    '''
    Calculate left_pixel at x - right_pixel at x+d
    '''
    return np.abs(int(limg[y][x])-int(rimg[y][x-d]))


# Census -----------------------------------------------------------------------
def censusTrans(img,w,h,x,y):
    '''
    Performs 3x3 census transform - only takes gray scale

        | 0 1 2 |
        | 3 p 4 |  ==> [ 0 1 2 3 4 5 6 7 ]
        | 5 7 7 |

    1 if < p else 0

    '''

    census = np.zeros(8)

    count = 0
    for yy in range(3):
        for xx in range(3):
            if (xx == 1 & yy == 1):
                continue

            elif (x-1+xx < 0) | (x+2+xx > w) | (y-1+yy < 0) | (y+1+yy > h) :
                count = count + 1
                continue

            if img[y-1+yy][x-1+xx] < img[y][x]:
                census[count] = 1
            count = count + 1

    return census

# Hamming distance -------------------------------------------------------------
def Hamm(arr1,arr2):
    '''
    My own hamming distance for 8 bits / parts
    '''
    hammDst = 0
    for i in range(8):
        if arr1[i] != arr2[i]:
            hammDst = hammDst + 1

    return hammDst

# permeability -----------------------------------------------------------------
def permeAggre(img,x,y):
    '''
    this function calculates permeability weights
    '''
    # parameters
    sigma = 0.5

    my_l = np.exp(-np.abs(int(img[y][x])-int(img[y][x-1]))/sigma)
    my_r = np.exp(-np.abs(int(img[y][x])-int(img[y][x+1]))/sigma)
    my_u = np.exp(-np.abs(int(img[y][x])-int(img[y-1][x]))/sigma)
    my_d = np.exp(-np.abs(int(img[y][x])-int(img[y+1][x]))/sigma)


    return my_l, my_r, my_u, my_d

def permeR(img,x,y,sigma=0.5):
    '''
    this function calculates permeability weight in right direction
    '''
    return np.exp(-np.abs(int(img[y][x])-int(img[y][x+1]))/sigma)

def permeL(img,x,y,sigma=0.5):
    '''
    this function calculates permeability weight in right direction
    '''
    return np.exp(-np.abs(int(img[y][x])-int(img[y][x-1]))/sigma)
    
def permeU(img,x,y,sigma=0.5):
    '''
    this function calculates permeability weight in right direction
    '''
    return np.exp(-np.abs(int(img[y][x])-int(img[y-1][x]))/sigma)

def permeD(img,x,y,sigma=0.5):
    '''
    this function calculates permeability weight in right direction
    '''
    return np.exp(-np.abs(int(img[y][x])-int(img[y+1][x]))/sigma)

# SWS right order - left to right ----------------------------------------------
def SWS_r(limg,rimg,x,y,d,para):
    '''
    This function will perform successive weighted summation

    sws in right direction: (y isn't present because of going horizontal)
    cost update:
    c^r_d (x) = C_comb(x) + my_r(x-1) * c^r_d(x-1)
    successive applying the cost update gives:
    c^r_d (x) = C_comb(x) + sum^(x-1)_(i=1) ( C_comb(x-i) * product^(i)_(j=1) my_r(x-j) )
        product^(i)_(j=1) my_r(x-j) = W^(r)_eff (x-i)
              = C_comb(x) + sum^(x-1)_(i=1) ( C_comb(x-i) * W^(r)_eff(x-i))

    '''
    C_sum = 0
    my_prod = 1

    ran = para['ran'] if para['ran'] < x-1 else x-1
    for i in range (ran):
        for ii in range(i):
            my_prod = my_prod * permeR(limg,x-(ii+1),y)

        C_sad = SADcost(limg,rimg,x-(i+1),y,d)
        C_ct = Hamm(censusTrans(limg,width,height,x-(i+1),y),
            censusTrans(rimg,width,height,x+d-(i+1),y))
        C_comb = para['alpha']*C_sad+(1-para['alpha'])*C_ct
        C_sum = C_sum + C_comb * my_prod

    C_sad = SADcost(limg,rimg,x,y,d)
    C_ct = Hamm(censusTrans(limg,width,height,x,y),
        censusTrans(rimg,width,height,x,y))
    C_comb = para['alpha']*C_sad+(1-para['alpha'])*C_ct

    return C_comb + C_sum

# SWS left order - right to left -----------------------------------------------
def SWS_l(limg,rimg,x,y,d,para):
    '''
    This function will perform successive weighted summation

    sws in left direction: (y isn't present because of going horizontal)
    cost update: (taken from right direction)
    c^r_d (x) = C_comb(x) + my_r(x-1) * c^r_d(x-1)
    successive applying the cost update gives:
    c^r_d (x) = C_comb(x) + sum^(x-1)_(i=1) ( C_comb(x-i) * product^(i)_(j=1) my_r(x-j) )
        product^(i)_(j=1) my_r(x-j) = W^(r)_eff (x-i)
              = C_comb(x) + sum^(x-1)_(i=1) ( C_comb(x-i) * W^(r)_eff(x-i))

    '''
    C_sum = 0
    my_prod = 1

    ran = para['ran'] if para['ran'] < limg.shape[1]-x-1 else limg.shape[1]-x-1
    for i in range (ran):
        for ii in range(i):
            my_prod = my_prod * permeL(limg,x+(ii+1),y)

        C_sad = SADcost(limg,rimg,x+(i+1),y,d)
        C_ct = Hamm(censusTrans(limg,width,height,x+(i+1),y),
            censusTrans(rimg,width,height,x+d+(i+1),y))
        C_comb = para['alpha']*C_sad+(1-para['alpha'])*C_ct
        C_sum = C_sum + C_comb * my_prod

    C_sad = SADcost(limg,rimg,x,y,d)
    C_ct = Hamm(censusTrans(limg,width,height,x,y),
        censusTrans(rimg,width,height,x,y))
    C_comb = para['alpha']*C_sad+(1-para['alpha'])*C_ct

    return C_comb + C_sum

# horizontal -------------------------------------------------------------------
def SWS_total(limg,rimg,x,y,d,para):
    '''
    This combines sws_r and sws_l
    '''
    swsl = SWS_l(limg,rimg,x,y,d,para)
    swsr = SWS_r(limg,rimg,x,y,d,para)
    swsu = SWS_u(limg,rimg,x,y,d,para)
    swsd = SWS_d(limg,rimg,x,y,d,para)
    
    return swsl + swsr + swsu + swsd

# SWS up order - down to up -----------------------------------------------
def SWS_u(limg,rimg,x,y,d,para):
    '''
    This function will perform successive weighted summation

    '''
    C_sum = 0
    my_prod = 1

    ran = para['ran'] if para['ran'] < limg.shape[0]-y-1 else limg.shape[0]-y-1
    for i in range (ran):
        for ii in range(i):
            my_prod = my_prod * permeU(limg,x,y+(ii+1))

        C_sad = SADcost(limg,rimg,x,y+(i+1),d)
        C_ct = Hamm(censusTrans(limg,width,height,x,y+(i+1)),
            censusTrans(rimg,width,height,x+d,y+(i+1)))
        C_comb = para['alpha']*C_sad+(1-para['alpha'])*C_ct
        C_sum = C_sum + C_comb * my_prod

    C_sad = SADcost(limg,rimg,x,y,d)
    C_ct = Hamm(censusTrans(limg,width,height,x,y),
        censusTrans(rimg,width,height,x,y))
    C_comb = para['alpha']*C_sad+(1-para['alpha'])*C_ct

    return C_comb + C_sum

# SWS down order - up to down -----------------------------------------------   
def SWS_d(limg,rimg,x,y,d,para):
    '''
    This function will perform successive weighted summation

    '''
    C_sum = 0
    my_prod = 1
    
    ran = para['ran'] if para['ran'] < y-1 else y-1
    for i in range (ran):
        for ii in range(i):
            my_prod = my_prod * permeU(limg,x,y-(ii+1))

        C_sad = SADcost(limg,rimg,x,y-(i+1),d)
        C_ct = Hamm(censusTrans(limg,width,height,x,y-(i+1)),
            censusTrans(rimg,width,height,x+d,y-(i+1)))
        C_comb = para['alpha']*C_sad+(1-para['alpha'])*C_ct
        C_sum = C_sum + C_comb * my_prod

    C_sad = SADcost(limg,rimg,x,y,d)
    C_ct = Hamm(censusTrans(limg,width,height,x,y),
        censusTrans(rimg,width,height,x,y))
    C_comb = para['alpha']*C_sad+(1-para['alpha'])*C_ct

    return C_comb + C_sum


# Main loop ###################################################################
if __name__ == "__main__" or True:
    
    start = time.time()
    # parameters
    para = {
        'alpha':    0.25,
        'ran':      1
    }

    # test parameters
    t = {
        'disp':     20,
        'x':        50,
        'y':        50
    }

    limg = read_pgm("../images/clq.pgm", byteorder='<')
    rimg = read_pgm("../images/crq.pgm", byteorder='<')

    height = limg.shape[0]
    width = limg.shape[1]

    # 4 steps -->
    #   1. cost calculation
    #   2. Aggregation
    #   3. Minimization
    #   4. Occlusion Handling

    # Cost calculation consist of SAD and Hamming and Census
    # SAD cost
    # C_sad = SADcost(limg,rimg,t['x'],t['y'],t['disp'])
    #
    # # census
    # censl = censusTrans(limg,width,height,t['x'],t['y'])
    # censr = censusTrans(rimg,width,height,t['x']-t['disp'],t['y'])
    # C_ct = Hamm(censl,censr)
    #
    # # combine cost:
    # combCost = para['alpha']*C_sad+(1-para['alpha'])*C_ct
    # print "sad:",C_sad,"hm:",C_ct,"=",combCost
    #
    # ## Aggregation
    #
    # # permeability:
    # my = permeAggre(limg,t['x'],t['y'])
        
    # SWS
    minval = 100
    
    dispmap = np.zeros((limg.shape[0],limg.shape[1]))
    for y in range(limg.shape[0]):
        for x in range(limg.shape[1]):
            minval = 100000
            for i in range(60):
                sws = SWS_total(limg,rimg,x,y,i,para)
                if sws < minval:
                    minval = sws
                    ii = i
            
            dispmap[y][x] = ii
        
        if y%10== 0:
            print y
                    
        


    print "min val", minval ,"at", ii 
    
    print "Script took", (time.time() - start), "secs"
