# EEPSM NEW implementation

import numpy as np
import time
from matplotlib import pyplot as plt
import sys
import cv2
from scipy import signal


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
    data = map(int, data)
    return np.asarray(data).reshape(int(size_y), int(size_x), 3)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# Calculate hamming distance --------------------------------------------------
def Hamm(arr1, arr2):
    '''
    My own hamming distance for 8 bits / parts
    '''
    hammDst = 0
    for i in range(8):
        if arr1[i] != arr2[i]:
            hammDst = hammDst + 1

    return hammDst


def censustrans_p(data, p_val=1, r=1):
    '''
    This function is an implementation of the census transform which can be
    performed on full images i.e. no need to cut up the image before. The size
    of the census transform window with the radius r

    args:
        data: matrix containing data
        p_val: value to pad border with
        r : radius for a square window
    returns:
        c_data_padded: matrix containing census data for each point in data and
                       is padded at the border
    '''

    # Get size of the data matrix
    m, n = data.shape

    # Generate matrix for census data
    if r == 1:
        c_data = np.zeros((m - (r * 2), n - (r * 2)), dtype='uint8')
        c_data_padded = np.ones((m, n), dtype='uint8') * p_val
    elif r == 2:
        c_data = np.zeros((m - (r * 2), n - (r * 2)), dtype='uint32')
        c_data_padded = np.ones((m, n), dtype='uint32') * p_val
    else:
        raise ValueError("A radius of", r, "is not supported")

    # Generate matrix corresponding to center pixels
    center_pixels = data[r:m - r, r:n - r]

    # Generate offsets for non-central pixels
    offsets = [(x, y) for y in range(2 * r + 1) for x in range(2 * r + 1)
               if not x == y == r]

    # Compare pixels
    for x, y in offsets:
        c_data = (c_data << 1) | (data[y:y + m - (r * 2), x:x + n - (r * 2)] >=
                                  center_pixels)

    c_data_padded[r:m - r, r:n - r] = c_data

    return c_data_padded


def hamming_dist_one(data1, data2):
    '''
    This function is an implementation of hamming distance which calculates the
    hamming distance between two uint8 values

    args:
        data1  - uint8 value 1
        data2  - uint8 value 2
    returns:
        h_data - hamming distance
    '''
    if data1.dtype == "uint8":
        l = 7
    elif data1.dtype == "uint32":
        l = 23

    return np.count_nonzero((data1 >> (range(l, -1, -1)) & 1) !=
                            (data2 >> (range(l, -1, -1)) & 1))


def hamming_dist(mat1, mat2):
    '''
    This function calls a calculation of hamming distance element wise between
    two matrices.

    args:
        mat1   - matrix 1
        mat2   - matrix 2
    returns:
        h_data - matrice which contains hamming distances for each element
    '''

    if np.shape(mat1) != np.shape(mat2):
        raise ValueError("Matrices dimensions must be equal")
    else:
        m, n = mat1.shape

    h_data = np.zeros((m, n))
    for y in range(m):
        for x in range(n):
            h_data[y, x] = hamming_dist_one(mat1[y, x], mat2[y, x])

    return h_data


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

            if data[y, x] < data[1, 1]:
                censustransform[count] = 1
                count += 1

    return censustransform


def permeability(img, x, y, sigma=0.5, direct='r'):
    '''
    this function calculates permeability weight in the specified direction
    and it uses color

    args:
        img: image data to calculate the weight from.
        x: x coordinate
        y: y coordinate
        sigma: smoothing factor - default: 0.5
        direct: Direction for the weight - default: right

    return:
        permeability weight for the specified direction
    '''
    if direct == 'r':
        # If the direction is right then:
        return np.min(np.exp(-np.abs(img[y, x] - img[y, x + 1]) / sigma))

    elif direct == 'l':
        # If the direction is left then:
        return np.min(np.exp(-np.abs(img[y, x] - img[y, x - 1]) / sigma))

    elif direct == 'b':
        # If the direction is top then:
        return np.min(np.exp(-np.abs(img[y, x] - img[y + 1, x]) / sigma))

    elif direct == 't':
        # If the direction is bottom then:
        return np.min(np.exp(-np.abs(img[y, x] - img[y - 1, x]) / sigma))
    else:
        # Error
        return None


def save_pfm(file, image, scale=1):
    '''
    Saves image to pfm file.

    Found at https://gist.github.com/chpatrick/8935738
    '''

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and \
            image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W'
                        'dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)

# main loop -------------------------------------------------------------------
if __name__ == "main" or True:

    start = time.time()
    print 'script started at', time.strftime("%H:%M:%S.", time.localtime())

    # filenames
    fdict = {'con': ['data/usable/conl.ppm', 'data/usable/conr.ppm', 59],
             'conf': ['data/usable/conlf.ppm', 'data/usable/conrf.ppm', 236],
             'ted': ['data/usable/tedl.ppm', 'data/usable/tedr.ppm', 59],
             'tedf': ['data/usable/tedlf.ppm', 'data/usable/tedrf.ppm', 236],
             'mot': ['data/usable/motl.ppm', 'data/usable/motr.ppm', 70],
             'tsu': ['data/usable/tsul.ppm', 'data/usable/tsur.ppm', 30]}

    # set constants
    image = 'mot'  # decide which stereo pair will be used.

    lim = 2  # limit used for consistancy check
    al = 0.4  # Weight for sad or census cost
    cr = 1  # Window radius for census transform
    cen_norm = 8 if cr == 1 else 24  # Value used for normalise census cost
    sig = 38.1  # Sigma value / Smoothing factor for permeability weights
    maxDisp = fdict[image][2]  # maximum disparity value in the
    # scene
    consistancy_check = True  # whether consistancy check should be preformed
    tB = 3.0 / 255.0  # Threshold value for border area
    L = 10   # number of sub-images used

    fnamel = fdict[image][0]  # Path to left image
    fnamer = fdict[image][1]  # Path to right image

    # load images and normalise the data
    Limg = readcolorppm(fnamel)  # Left image
    Rimg = readcolorppm(fnamer)  # Right image

    LimgG = rgb2gray(Limg)  # Left image grayscale
    RimgG = rgb2gray(Rimg)  # Right image grayscale

    # Divid original images into sub images
    newY = np.ceil(Limg.shape[0] / (1.0 * L))
    LimgSub = np.zeros((L, newY, Limg.shape[1], Limg.shape[2]))
    RimgSub = np.zeros((L, newY, Limg.shape[1], Limg.shape[2]))
    LimgGSub = np.zeros((L, newY, Limg.shape[1]))
    RimgGSub = np.zeros((L, newY, Limg.shape[1]))

    start = time.time()
    for sub in range(L):
        if Limg[sub * newY:].shape[0] < newY:
            tmp_val = Limg[sub * newY:].shape[0]
            LimgSub[sub, :tmp_val] = Limg[sub * newY:(sub + 1) * newY]
            RimgSub[sub, :tmp_val] = Rimg[sub * newY:(sub + 1) * newY]
            LimgGSub[sub, :tmp_val] = LimgG[sub * newY:(sub + 1) * newY]
            RimgGSub[sub, :tmp_val] = RimgG[sub * newY:(sub + 1) * newY]
        else:
            LimgSub[sub] = Limg[sub * newY:(sub + 1) * newY]
            RimgSub[sub] = Rimg[sub * newY:(sub + 1) * newY]
            LimgGSub[sub] = LimgG[sub * newY:(sub + 1) * newY]
            RimgGSub[sub] = RimgG[sub * newY:(sub + 1) * newY]

    # Image to contain the final disparity map
    final_disp_map = np.zeros((newY * L, Limg.shape[1]))

    for sub in range(L):
        print sub

        Limg = LimgSub[sub]
        Rimg = RimgSub[sub]
        LimgG = LimgGSub[sub]
        RimgG = RimgGSub[sub]

        # mirrored along the x-axis versions
        Limg_m = Limg[:, ::-1]   # Left image mirrored
        Rimg_m = Rimg[:, ::-1]   # Right image mirroed
        LimgG_m = LimgG[:, ::-1]  # Left image grayscale mirrored
        RimgG_m = RimgG[:, ::-1]  # Right image grayscale mirrored

        # generate census transform for grayscale images
        Lcen = censustrans_p(LimgG, 0, cr)
        Rcen = censustrans_p(RimgG, 7, cr)
        Lcen_m = censustrans_p(LimgG_m, 0, cr)  # mirrored
        Rcen_m = censustrans_p(RimgG_m, 7, cr)  # mirrored

        # Generate 4 matrices which can contain Aggregation for the 4
        # directions
        # Going from right to left
        aggre_L = np.zeros(LimgG.shape + (maxDisp,))
        # Going from left to right
        aggre_R = np.zeros(LimgG.shape + (maxDisp,))
        # Going from bottom to top
        aggre_T = np.zeros(LimgG.shape + (maxDisp,))
        # Going from top to bottom
        aggre_B = np.zeros(LimgG.shape + (maxDisp,))

        # Generate variables
        m, n, c = Limg.shape  # Size variables
        # Disparity estimates cost values
        dispCost = np.ones((m, n, maxDisp)) * tB
        # Mirrored disp. est. cost val.
        dispCost_m = np.ones((m, n, maxDisp)) * tB

        # Calculate initial cost
        for d in range(maxDisp):
            # calculate SAD
            # Generate temp. var. for shifted right
            tmp = np.ones((m, n, c)) * tB
            # image values
            # Shift right image values to the
            tmp[:, d:n, :] = Rimg[:, 0:n - d, :]
            # left depending on the current disparity value.
            c_color = np.abs(tmp - Limg) * 0.333333  # Calculate diff.
            c_color = np.sum(c_color, 2)  # Sum diff
            c_color = c_color / np.max(c_color)  # Normalise values

            # calculate census transform
            tmp = np.ones((m, n), dtype='uint8')  # Generate temp. var for
            # shifted right image census values
            # Shift values by the current disparity
            tmp[:, d:n] = Rcen[:, 0:n - d]
            c_cen = hamming_dist(Lcen, tmp) / cen_norm  # Normalise values

            # calculate total cost
            c_tot = al * c_color + (1 - al) * c_cen  # Combine costs

            # do the same for the other view
            # calculate SAD
            tmp_m = np.ones((m, n, c)) * tB  # Generate temp. var. for shifted
            # right image values
            # Shift left image values depending
            tmp_m[:, d:n] = Limg_m[:, 0:n - d]
            # on the current disparity value.
            c_color_m = np.abs(tmp_m - Rimg_m) * 0.333333  # Calculate diff.
            c_color_m = np.sum(c_color_m, 2)  # Sum diff.
            c_color_m = c_color_m / np.max(c_color_m)  # Normalise values

            # census transform
            tmp_m = np.ones((m, n), dtype='uint8')  # Generate temp. var for
            # shifted left image census values
            tmp_m[:, d:n] = Lcen_m[:, 0:n - d]  # Shift values by the current
            # disparity
            c_cen_m = hamming_dist(Rcen_m, tmp_m) / \
                cen_norm  # Normalise values

            # total
            c_tot_m = al * c_color_m + (1 - al) * c_cen_m  # Combine costs

            # set values
            dispCost[:, :, d] = c_tot
            dispCost_m[:, :, d] = c_tot_m

        costMat = dispCost
        costMat_m = dispCost_m

        # generate matrices for permeability weights
        perme_l = np.zeros(LimgG.shape)
        perme_r = np.zeros(LimgG.shape)
        perme_t = np.zeros(LimgG.shape)
        perme_b = np.zeros(LimgG.shape)

        # calculate permeability weights
        for y in range(Limg.shape[0]):
            for x in range(Limg.shape[1]):
                if y == Limg.shape[0] - 1:
                    perme_t[y, x] = permeability(
                        Limg, x, y, sigma=sig, direct='t')
                    perme_b[y, x] = 0
                elif y == 0:
                    perme_t[y, x] = 0
                    perme_b[y, x] = permeability(
                        Limg, x, y, sigma=sig, direct='b')
                else:
                    perme_t[y, x] = permeability(
                        Limg, x, y, sigma=sig, direct='t')
                    perme_b[y, x] = permeability(
                        Limg, x, y, sigma=sig, direct='b')

                if x == 0:
                    perme_l[y, x] = 0
                    perme_r[y, x] = permeability(
                        Limg, x, y, sigma=sig, direct='r')
                elif x == Limg.shape[1] - 1:
                    perme_l[y, x] = permeability(
                        Limg, x, y, sigma=sig, direct='l')
                    perme_r[y, x] = 0
                else:
                    perme_l[y, x] = permeability(
                        Limg, x, y, sigma=sig, direct='l')
                    perme_r[y, x] = permeability(
                        Limg, x, y, sigma=sig, direct='r')

        # Left scan value calculation
        aggre_L[:, Limg.shape[1] - 1, :] = costMat[:, Limg.shape[1] - 1, :]
        for x in range(Limg.shape[1] - 2, 0, -1):
            for d in range(maxDisp):
                aggre_L[:, x, d] = costMat[:, x, d] + perme_l[:, x + 1] * \
                    aggre_L[:, x + 1, d]

        # Right scan value calculation
        aggre_R[:, 0, :] = costMat[:, 0, :]
        for x in range(1, Limg.shape[1] - 1):
            for d in range(maxDisp):
                aggre_R[:, x, d] = costMat[:, x, d] + perme_r[:, x - 1] * \
                    aggre_R[:, x - 1, d]

        # horizontal aggregation
        horz_aggre = np.zeros(LimgG.shape + (maxDisp,))
        for x in range(0, Limg.shape[1] - 1):
            for d in range(maxDisp):
                horz_aggre[:, x, d] = aggre_R[:, x, d] + aggre_L[:, x, d]

        # top scan value calculation
        aggre_T[-1, :, d] = horz_aggre[-1, :, d]
        for y in range(Limg.shape[0] - 2, 0, -1):
            for d in range(maxDisp):
                aggre_T[y, :, d] = horz_aggre[y, :, d] + perme_t[y + 1, :] * \
                    aggre_T[y + 1, :, d]

        # new bottom scan value calculation
        aggre_B[0, :, d] = horz_aggre[0, :, d]
        for y in range(1, LimgG.shape[0] - 1):
            for d in range(maxDisp):
                aggre_B[y, :, d] = horz_aggre[y, :, d] + perme_b[y - 1, :] * \
                    aggre_B[y - 1, :, d]

        # combine vertical aggregation data
        total_aggre = np.zeros(LimgG.shape + (maxDisp,))
        for y in range(0, Limg.shape[0] - 1):
            for d in range(maxDisp):
                total_aggre[y, :, d] = aggre_T[y, :, d] + aggre_B[y, :, d]

        # generate disparity map
        dispmap = np.zeros(RimgG.shape)
        for y in range(Limg.shape[0]):
            for x in range(Limg.shape[1]):
                dispmap[y, x] = np.argmin(total_aggre[y, x])

        # perform consistancy check and find and fill occlusions
        if consistancy_check:

            # calculate permeability weights
            for y in range(Rimg_m.shape[0]):
                for x in range(Rimg_m.shape[1]):
                    if y == Rimg.shape[0] - 1:
                        perme_t[y, x] = permeability(
                            Rimg_m, x, y, sigma=sig, direct='t')
                        perme_b[y, x] = 0
                    elif y == 0:
                        perme_t[y, x] = 0
                        perme_b[y, x] = permeability(
                            Rimg_m, x, y, sigma=sig, direct='b')
                    else:
                        perme_t[y, x] = permeability(
                            Rimg_m, x, y, sigma=sig, direct='t')
                        perme_b[y, x] = permeability(
                            Rimg_m, x, y, sigma=sig, direct='b')

                    if x == 0:
                        perme_l[y, x] = 0
                        perme_r[y, x] = permeability(
                            Rimg_m, x, y, sigma=sig, direct='r')
                    elif x == Rimg.shape[1] - 1:
                        perme_l[y, x] = permeability(
                            Rimg_m, x, y, sigma=sig, direct='l')
                        perme_r[y, x] = 0
                    else:
                        perme_l[y, x] = permeability(
                            Rimg_m, x, y, sigma=sig, direct='l')
                        perme_r[y, x] = permeability(
                            Rimg_m, x, y, sigma=sig, direct='r')

            # Left scan value calculation
            aggre_L[:, Rimg_m.shape[1] - 1,
                    :] = costMat_m[:, Limg.shape[1] - 1, :]
            for x in range(Limg.shape[1] - 2, 0, -1):
                for d in range(maxDisp):
                    aggre_L[:, x, d] = costMat_m[:, x, d] + perme_l[:, x + 1] \
                        * aggre_L[:, x + 1, d]

            # Right scan value calculation
            aggre_R[:, 0, :] = costMat_m[:, 0, :]
            for x in range(1, Limg.shape[1] - 1):
                for d in range(maxDisp):
                    aggre_R[:, x, d] = costMat_m[:, x, d] + perme_r[:, x - 1] \
                        * aggre_R[:, x - 1, d]

            # horizontal aggregation
            horz_aggre = np.zeros(RimgG.shape + (maxDisp,))
            for x in range(0, Limg.shape[1] - 1):
                for d in range(maxDisp):
                    horz_aggre[:, x, d] = aggre_R[:, x, d] + aggre_L[:, x, d]

            # top scan value calculation
            aggre_T[Rimg.shape[0] - 1, :,
                    d] = horz_aggre[Limg.shape[0] - 1, :, d]
            for y in range(Limg.shape[0] - 2, 0, -1):
                for d in range(maxDisp):
                    aggre_T[y, :, d] = horz_aggre[y, :, d] + perme_t[y + 1, :]\
                        * aggre_T[y + 1, :, d]

            # new bottom scan value calculation
            aggre_B[0, :, d] = horz_aggre[0, :, d]
            for y in range(1, LimgG.shape[0] - 1):
                for d in range(maxDisp):
                    aggre_B[y, :, d] = horz_aggre[y, :, d] + perme_b[y - 1, :]\
                        * aggre_B[y - 1, :, d]

            # combine vertical aggregation data
            total_aggre = np.zeros(LimgG.shape + (maxDisp,))
            for y in range(0, Limg.shape[0] - 1):
                for d in range(maxDisp):
                    total_aggre[y, :, d] = aggre_T[y, :, d] + aggre_B[y, :, d]

            total_aggre = total_aggre[:, ::-1, :]

            # generate disparity map
            dispmap2 = np.zeros(RimgG.shape)
            for y in range(Limg.shape[0]):
                for x in range(Limg.shape[1]):
                    dispmap2[y, x] = np.argmin(total_aggre[y, x])

            # find and fill occlusions - nda
            dispmap_final = 1 * dispmap
            m, n = dispmap.shape
            for y in range(m):
                for x in range(n):
                    if np.abs(dispmap[y, x] - dispmap2[y, x - dispmap[y, x]]) \
                            >= lim:
                        dispmap_final[y, x] = -1

            # fill occluded points - nda
            dispmap_final_filled = 1 * dispmap_final
            for y in range(m):
                for x in range(maxDisp, n):
                    if dispmap_final_filled[y, x] <= 0:
                        dispmap_final_filled[
                            y, x] = dispmap_final_filled[y, x - 1]
            for y in range(m):  # bord
                for x in range(maxDisp + 10, 0, -1):
                    if dispmap_final_filled[y, x] <= 0:
                        dispmap_final_filled[
                            y, x] = dispmap_final_filled[y, x + 1]

        final_disp_map[sub * newY: (sub + 1) * newY] = dispmap_final_filled

    print 'It took', time.time() - start

    for sub in range(L):
        final_disp_map[sub * newY - 1] = final_disp_map[sub * newY - 2]
        final_disp_map[sub * newY] = final_disp_map[sub * newY + 1]

    dispmap_final_filled = final_disp_map
    # # save the out as .png for the report
    plt.figure()
    fstr = 'data/res/' + image + '_cr' + str(cr) + '_s' + str(sig) + '_al' + \
        str(al) + '_eepsm_div.png'
    plt.imsave(fstr, dispmap_final_filled, cmap=plt.cm.gray)
    print "image saved as:" + fstr

    plt.close()

    if consistancy_check:
        plt.figure()
        plt.imshow(dispmap_final_filled, cmap=plt.cm.gray)

    # save the ouput as .pfm if pfm files exist for the images
    imgList = ['mot', 'ted']
    if any(image in s for s in imgList):
        fstr = 'data/res/' + image + '_eepsm_div.pfm'
        file = open(fstr, 'wb')
        save_pfm(file, dispmap_final_filled.astype('float32'), scale=1)
        print "image saved as:" + fstr
        file.close()

    plt.show()
