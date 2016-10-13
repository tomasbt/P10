# EEPSM NEW implementation

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

            if data[y][x] < data[1][1]:
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
        return np.min(np.exp(-np.abs(img[y][x] - img[y][x + 1]) / sigma))

    elif direct == 'l':
        # If the direction is left then:
        return np.min(np.exp(-np.abs(img[y][x] - img[y][x - 1]) / sigma))

    elif direct == 't':
        # If the direction is top then:
        return np.min(np.exp(-np.abs(img[y][x] - img[y + 1][x]) / sigma))

    elif direct == 'b':
        # If the direction is bottom then:
        return np.min(np.exp(-np.abs(img[y][x] - img[y - 1][x]) / sigma))
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
    print 'script started at', time.strftime("%H:%M:%S.", time.localtime()), \
        'Expected done at', time.strftime("%H:%M.", time.localtime(
            time.time() + 300))

    # filenames
    fdict = {'con': ['data/usable/conl.ppm', 'data/usable/conr.ppm', 59],
             'conf': ['data/usable/conlf.ppm', 'data/usable/conrf.ppm', 236],
             'ted': ['data/usable/tedl.ppm', 'data/usable/tedr.ppm', 59],
             'tedf': ['data/usable/tedlf.ppm', 'data/usable/tedrf.ppm', 236],
             'mot': ['data/usable/motl.ppm', 'data/usable/motr.ppm', 70],
             'tsu': ['data/usable/tsul.ppm', 'data/usable/tsur.ppm', 30],
             'nku': ['data/usable/nkul.ppm', 'data/usable/nkur.ppm', 130],
             'ven': ['data/usable/venl.ppm', 'data/usable/venr.ppm', 32]}

    # set constants
    image = 'tsu'  # decide which stereo pair will be used.

    lim = 2
    al = 0.5
    cr = 2
    cen_norm = 8 if cr == 1 else 24
    sig = 15
    maxDisp = fdict[image][2]
    consistancy_check = False
    tB = 3.0 / 255.0
    tCo = 7.0 / 255.0
    tCe = 2.0 / 255.0

    fnamel = fdict[image][0]
    fnamer = fdict[image][1]

    # load images and normalise the data
    Limg = readcolorppm(fnamel)  # /255.0  # Left image
    Rimg = readcolorppm(fnamer)  # /255.0  # Right image
    LimgG = rgb2gray(Limg)            # Left image grayscale
    RimgG = rgb2gray(Rimg)            # Right image grayscale
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

    # Generate 4 matrices which can contain Aggregation for the 4 directions
    aggre_L = np.zeros(LimgG.shape + (maxDisp,))  # Going from right to left
    aggre_R = np.zeros(LimgG.shape + (maxDisp,))  # Going from left to right
    aggre_T = np.zeros(LimgG.shape + (maxDisp,))  # Going from bottom to top
    aggre_B = np.zeros(LimgG.shape + (maxDisp,))  # Going from top to bottom

    # Generate variables
    m, n, c = Limg.shape  # Size variables
    dispCost = np.ones((m, n, maxDisp)) * tB  # Disparity estimates cost values
    dispCost_m = np.ones((m, n, maxDisp)) * tB  # Mirrored disp. est. cost val.

    start = time.time()
    # Calculate initial cost
    for d in range(maxDisp):
        # calculate SAD
        tmp = np.ones((m, n, c)) * tB
        tmp[:, d:n, :] = Rimg[:, 0:n - d, :]
        c_color = np.abs(tmp - Limg) * 0.333333
        c_color = np.sum(c_color, 2)
        c_color = c_color / np.max(c_color)

        # calculate census transform
        tmp = np.ones((m, n), dtype='uint8')
        tmp[:, d:n] = Rcen[:, 0:n - d]
        c_cen = hamming_dist(Lcen, tmp) / cen_norm

        # calculate total cost
        c_tot = al * c_color + (1 - al) * c_cen

        # do the same for the other view
        # SAD
        tmp_m = np.ones((m, n, c)) * tB
        tmp_m[:, d:n] = Limg_m[:, 0:n - d]
        c_color_m = np.abs(tmp_m - Rimg_m) * 0.333333
        c_color_m = np.sum(c_color_m, 2)
        c_color_m = c_color_m / np.max(c_color_m)

        # census transform
        tmp_m = np.ones((m, n), dtype='uint8')
        tmp_m[:, d:n] = Lcen_m[:, 0:n - d]
        c_cen_m = hamming_dist(Rcen, tmp_m) / cen_norm

        # total
        c_tot_m = al * c_color_m + (1 - al) * c_cen_m

        # set values
        dispCost[:, :, d] = c_tot
        dispCost_m[:, :, d] = c_tot_m[:, ::-1]

    print "new cost takes", time.time() - start, "seconds"
    # start = time.time()
    # # calculate cost for each pixel
    # costMat = np.zeros(RimgG.shape + (maxDisp,))
    # for y in range(1, Limg.shape[0] - 1):
    #     for x in range(1, Limg.shape[1] - 1):
    #         maxD = maxDisp if x - maxDisp >= 0 else x
    #         for d in range(maxD):
    #             costMat[y][x][d] = al * np.sum(
    #                 np.abs(Limg[y][x] - Rimg[y][x - d])) + \
    #                 (1 - al) * Hamm(ctrans(LimgG[y - 1:y + 2, x - 1:x + 2]),
    #                                 ctrans(RimgG[y - 1:y + 2,
    #                                              x - 1 - d:x + 2 - d]))
    #
    # print "old cost takes", time.time() - start, "seconds"

    costMat = dispCost

    # generate matrices for permeability weights
    perme_l = np.zeros(LimgG.shape)
    perme_r = np.zeros(LimgG.shape)
    perme_t = np.zeros(LimgG.shape)
    perme_b = np.zeros(LimgG.shape)

    # calculate permeability weights
    for y in range(Limg.shape[0]):
        for x in range(Limg.shape[1]):
            if y == 0:
                perme_t[y, x] = permeability(Limg, x, y, sigma=sig, direct='t')
                perme_b[y, x] = 0
            elif y == Limg.shape[0] - 1:
                perme_t[y, x] = 0
                perme_b[y, x] = permeability(Limg, x, y, sigma=sig, direct='b')
            else:
                perme_t[y, x] = permeability(Limg, x, y, sigma=sig, direct='t')
                perme_b[y, x] = permeability(Limg, x, y, sigma=sig, direct='b')

            if x == 0:
                perme_l[y, x] = 0
                perme_r[y, x] = permeability(Limg, x, y, sigma=sig, direct='r')
            elif x == Limg.shape[1] - 1:
                perme_l[y, x] = permeability(Limg, x, y, sigma=sig, direct='l')
                perme_r[y, x] = 0
            else:
                perme_l[y, x] = permeability(Limg, x, y, sigma=sig, direct='l')
                perme_r[y, x] = permeability(Limg, x, y, sigma=sig, direct='r')

    # new left scan value calculation
    for x in range(Limg.shape[1] - 1, 0, -1):
        for d in range(maxDisp):
            aggre_L[:, x, d] = costMat[:, x, d] + perme_r[:, x] * \
                aggre_L[:, x - 1, d]

    # new right scan value calculation
    for x in range(1, Limg.shape[1] - 1):
        for d in range(maxDisp):
            aggre_R[:, x, d] = costMat[:, x, d] + perme_l[:, x] * \
                aggre_R[:, x - 1, d]

    # # Calculate left scan values
    # for y in range(Limg.shape[0]):
    #     for x in range(Limg.shape[1], 0, -1):
    #         maxD = maxDisp if x - maxDisp >= 0 else x
    #         for d in range(maxD):
    #             if x == 0:
    #                 aggre_L[y][x][d] = costMat[y][x][d]
    #                 continue
    #             elif x >= Limg.shape[1] - 1:
    #                 continue
    #             # if y == 0:
    #             #     continue
    #             # elif y == Limg.shape[0] - 1:
    #             #     continue
    #
    #             aggre_L[y][x][d] = costMat[y][x][d] + \
    #                 permeability(LimgG, x, y, sigma=sig, direct='r') * \
    #                 aggre_L[y][x + 1][d]

    # horizontal aggregation
    horz_aggre = np.zeros(LimgG.shape + (maxDisp,))
    for y in range(1, Limg.shape[0] - 1):
        for x in range(0, Limg.shape[1] - 2):
            maxD = maxDisp if x - maxDisp >= 0 else x
            for d in range(maxD):
                # if x == 0:
                #     aggre_R[x][y][d] = costMat[y][x][d]
                #     continue
                # elif x >= Limg.shape[1] - 2:
                #     continue
                # if y == 0:
                #     continue
                # elif y >= Limg.shape[0] - 2:
                #     continue
                # aggre_R[y][x][d] = costMat[y][x][d] + \
                #     permeability(LimgG, x, y, sigma=sig, direct='l') * \
                #     aggre_R[y][x - 1][d]
                # print y, x, d
                horz_aggre[:, x, d] = costMat[:, x, d] + \
                    perme_l[:, x] * aggre_R[:, x - 1, d] + \
                    perme_r[:, x] * aggre_L[:, x + 1, d]

    # new right scan value calculation
    for x in range(1, Limg.shape[1] - 1):
        for d in range(maxDisp):
            aggre_R[:, x, d] = costMat[:, x, d] + perme_r[:, x] * \
                aggre_R[:, x - 1, d]

    # calculate top scan values - bottom to top
    for y in range(Limg.shape[0] - 1, 0, -1):
        for x in range(Limg.shape[1]):
            maxD = maxDisp if x - maxDisp >= 0 else x
            for d in range(maxD):
                if x == 0:
                    continue
                elif x >= Limg.shape[1] - 1:
                    continue
                if y == 0:
                    continue
                elif y >= Limg.shape[0] - 1:
                    aggre_T[y][x][d] = horz_aggre[y][x][d]
                    continue

                aggre_T[y][x][d] = horz_aggre[y][x][d] + \
                    permeability(LimgG, x, y, sigma=sig, direct='b') * \
                    aggre_T[y + 1][x][d]

    # combine vertical aggregation data
    total_aggre = np.zeros(LimgG.shape + (maxDisp,))
    for y in range(0, Limg.shape[0]):
        for x in range(0, Limg.shape[1]):
            maxD = maxDisp if x - maxDisp >= 0 else x
            for d in range(maxD):
                if x == 0:
                    continue
                elif x >= Limg.shape[1] - 1:
                    continue
                if y == 0:
                    aggre_B[y][x][d] = horz_aggre[y][x][d]
                    continue
                elif y >= Limg.shape[0] - 1:
                    continue

                aggre_B[y][x][d] = horz_aggre[y][x][d] + \
                    permeability(LimgG, x, y, sigma=sig, direct='t') * \
                    aggre_B[y - 1][x][d]
                total_aggre[y][x][d] = horz_aggre[y][x][d] + \
                    permeability(LimgG, x, y, sigma=sig, direct='t') * \
                    aggre_T[y + 1][x][d] + \
                    permeability(LimgG, x, y, sigma=sig, direct='b') * \
                    aggre_B[y - 1][x][d]

    # generate disparity map
    dispmap = np.zeros(RimgG.shape)
    for y in range(Limg.shape[0]):
        for x in range(Limg.shape[1]):
            dispmap[y][x] = np.argmin(total_aggre[y][x])

    if consistancy_check:
        # calculate cost for each pixel
        costMat = np.zeros(RimgG.shape + (maxDisp,))
        for y in range(1, Rimg.shape[0] - 1):
            for x in range(1, Rimg.shape[1] - 1):
                # maxD = maxDisp if x-maxDisp >= 0 else x
                maxD = maxDisp if x + maxDisp <= Rimg.shape[1] - 1 else \
                    Rimg.shape[1] - x - 1
                for d in range(maxD):
                    costMat[y][x][d] = al * np.sum(
                        np.abs(Rimg[y][x] - Limg[y][x + d])) + \
                        (1 - al) * Hamm(
                        ctrans(LimgG[y - 1:y + 2, x - 1:x + 2]),
                        ctrans(RimgG[y - 1:y + 2, x - 1 + d:x + 2 + d]))

        # Calculate left scan values
        for y in range(Rimg.shape[0]):
            for x in range(Rimg.shape[1], 0, -1):
                # maxD = maxDisp if x-maxDisp >= 0 else x
                maxD = maxDisp if x + maxDisp <= Limg.shape[1] - 1 else \
                    Limg.shape[1] - x - 1
                for d in range(maxD):
                    if x == 0:
                        continue
                    elif x >= Rimg.shape[1] - 1:
                        continue
                    if y == 0:
                        continue
                    elif y == Rimg.shape[0] - 1:
                        continue

                    aggre_L[y][x][d] = costMat[y][x][d] + \
                        permeability(RimgG, x, y, sigma=sig, direct='l') * \
                        aggre_L[y][x + 1][d]

        # horizontal aggregation
        horz_aggre = np.zeros(RimgG.shape + (maxDisp,))
        for y in range(1, Rimg.shape[0] - 1):
            for x in range(0, Rimg.shape[1] - 2):
                # maxD = maxDisp if x-maxDisp >= 0 else x
                maxD = maxDisp if x + maxDisp <= Limg.shape[1] - 1 else \
                    Limg.shape[1] - x - 1
                for d in range(maxD):
                    if x == 0:
                        aggre_R[x][y][d] = costMat[y][x][d]
                        continue
                        horz_aggre[y][x][d] = costMat[y][x][d] + \
                            permeability(RimgG, x + 1, y, sigma=sig,
                                         direct='l') * aggre_L[y][x + 1][d]
                        continue
                    elif x >= Rimg.shape[1] - 2:
                        continue
                    if y == 0:
                        continue
                    elif y >= Rimg.shape[0] - 2:
                        continue
                    aggre_R[y][x][d] = costMat[y][x][d] + \
                        permeability(RimgG, x, y, sigma=sig, direct='r') * \
                        aggre_R[y][x - 1][d]

                    horz_aggre[y][x][d] = costMat[y][x][d] + \
                        permeability(RimgG, x, y, sigma=sig, direct='l') * \
                        aggre_R[y][x - 1][d] + \
                        permeability(RimgG, x, y, sigma=sig, direct='r') * \
                        aggre_L[y][x + 1][d]

        # calculate top scan values - bottom to top
        for y in range(Rimg.shape[0] - 1, 0, -1):
            for x in range(Rimg.shape[1]):
                maxD = maxDisp if x - maxDisp >= 0 else x
                for d in range(maxD):
                    if x == 0:
                        continue
                    elif x >= Rimg.shape[1] - 1:
                        continue
                    if y == 0:
                        continue
                    elif y >= Rimg.shape[0] - 1:
                        aggre_T[y][x][d] = horz_aggre[y][x][d]
                        continue

                    aggre_T[y][x][d] = horz_aggre[y][x][d] + \
                        permeability(RimgG, x, y, sigma=sig, direct='t') * \
                        aggre_T[y + 1][x][d]

        # combine vertical aggregation data
        total_aggre = np.zeros(RimgG.shape + (maxDisp,))
        for y in range(0, Rimg.shape[0]):
            for x in range(0, Rimg.shape[1]):
                maxD = maxDisp if x - maxDisp >= 0 else x
                for d in range(maxD):
                    if x == 0:
                        continue
                    elif x >= Rimg.shape[1] - 1:
                        continue
                    if y == 0:
                        aggre_B[y][x][d] = horz_aggre[y][x][d]
                        continue
                    elif y >= Rimg.shape[0] - 1:
                        continue

                    aggre_B[y][x][d] = horz_aggre[y][x][d] + \
                        permeability(RimgG, x, y, sigma=sig, direct='t') * \
                        aggre_B[y - 1][x][d]
                    total_aggre[y][x][d] = horz_aggre[y][x][d] + \
                        permeability(RimgG, x, y, sigma=sig, direct='b') * \
                        aggre_T[y + 1][x][d] + \
                        permeability(RimgG, x, y, sigma=sig, direct='t') * \
                        aggre_B[y - 1][x][d]
        # generate disparity map
        dispmap2 = np.zeros(RimgG.shape)
        for y in range(Rimg.shape[0]):
            for x in range(Rimg.shape[1]):
                dispmap2[y][x] = np.argmin(total_aggre[y][x])

        # find and fill occlusions - nda
        dispmap_final = 1 * dispmap
        m, n = dispmap.shape
        for y in range(m):
            for x in range(n):
                if np.abs(dispmap[y][x] - dispmap2[y][x - dispmap[y][x]]) \
                        >= lim:
                    dispmap_final[y][x] = -1

        # fill occluded points - nda
        dispmap_final_filled = 1 * dispmap_final
        for y in range(m):
            for x in range(n):
                if dispmap_final_filled[y][x] <= 0:
                    dispmap_final_filled[y][x] = dispmap_final_filled[y][x - 1]
        for y in range(m):  # bord
            for x in range(maxDisp, 0, -1):
                if dispmap_final_filled[y][x] <= 0:
                    dispmap_final_filled[y][x] = dispmap_final_filled[y][x + 1]

    print 'It took', time.time() - start
    # save the out as .png for the report
    # plt.figure()
    # fstr = 'data/res/'+image+'_eepsm_1.png'
    # plt.imsave(fstr,dispmap,cmap=plt.cm.gray)
    # print "image saved as:" + fstr
    # plt.close()
    #
    # save the out as .png for the report
    plt.figure()
    fstr = 'data/res/' + image + '_cr' + str(cr) + '_s' + str(sig) + '_al' + \
        str(al) + '_eepsm_2.png'
    plt.imsave(fstr, dispmap, cmap=plt.cm.gray)
    print "image saved as:" + fstr

    plt.close()
    #

    plt.figure()
    plt.imshow(dispmap, cmap=plt.cm.gray)

    if consistancy_check:
        plt.figure()
        plt.imshow(dispmap2, cmap=plt.cm.gray)
    # # save the ouput as .pfm if pfm files exist for the images
    # if image == 'mot':
    #     fstr = 'data/res/'+image+'_eepsm.pfm'
    #     file = open(fstr,'wb')
    #     save_pfm(file, dispmap_final_filled.astype('float32'), scale = 1)
    #     file.close()
