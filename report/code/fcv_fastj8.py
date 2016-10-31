# fast fast cost volume simulation fcv.py

# imports
import numpy as np
import time
from matplotlib import pyplot as plt
import cv2


# functions
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


def save_pfm(file, image, scale=1):
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    # greyscale
    elif len(image.shape) == 2 or len(image.shape) == 3 and \
            image.shape[2] == 1:
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

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
    fdict = {'con': ['data/usable/conl.ppm', 'data/usable/conr.ppm', 59],
             'conf': ['data/usable/conlf.ppm', 'data/usable/conrf.ppm', 236],
             'ted': ['data/usable/tedl.ppm', 'data/usable/tedr.ppm', 59],
             'tedf': ['data/usable/tedlf.ppm', 'data/usable/tedrf.ppm', 236],
             'mot': ['data/usable/motl.ppm', 'data/usable/motr.ppm', 70],
             'tsu': ['data/usable/tsul.ppm', 'data/usable/tsur.ppm', 30],
             'nku': ['data/usable/nkul.ppm', 'data/usable/nkur.ppm', 130],
             'ven': ['data/usable/venl.ppm', 'data/usable/venr.ppm', 32],
             'art': ['data/usable/artl.ppm', 'data/usable/artr.ppm', 55],
             'pla': ['data/usable/plal.ppm', 'data/usable/plar.ppm', 150]}

    # set constants
    image = 'art'
    al = 0.11

    maxDisp = fdict[image][2]
    r = 9
    eps = 0.0001
    lim = 2
    tB = 3.0 / 255
    tC = 7.0 / 255
    tG = 2.0 / 255
    g_c = 0.1
    g_d = 9
    r_median = 19

    fnamel = fdict[image][0]
    fnamer = fdict[image][1]

    # load images and normalise the data
    Il = readcolorppm(fnamel) / 255.0
    Ir = readcolorppm(fnamer) / 255.0

    Limg = cv2.imread(fnamel)
    Rimg = cv2.imread(fnamer)
    Limg = cv2.normalize(Limg, None, 0.0, 1.0,
                         cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    Rimg = cv2.normalize(Rimg, None, 0.0, 1.0,
                         cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    Ilg = rgb2gray(Il)
    Irg = rgb2gray(Ir)

    # mirror images
    Il_1 = Il[:, ::-1, :]
    Ir_1 = Ir[:, ::-1, :]

    Rimg_1 = Rimg[:, ::-1, :]
    Limg_1 = Limg[:, ::-1, :]

    # compute gradient in X-direction
    fx_l = np.gradient(Ilg)[1]
    fx_r = np.gradient(Irg)[1]

    fx_l_1 = fx_l[:, ::-1]
    fx_r_1 = fx_r[:, ::-1]

    m, n, c = Il.shape

    dispVol = np.ones((m, n, maxDisp)) * tB
    dispVol1 = np.ones((m, n, maxDisp)) * tB

    print 'Starting cost calculation. Time taken so far', time.time() - start,\
        'seconds'

    for d in range(maxDisp):
        # calculate SAD
        tmp = np.ones((m, n, c)) * tB
        tmp[:, d:n, :] = Ir[:, 0:n - d, :]
        c_color = np.abs(tmp - Il)
        c_color = np.sum(c_color, 2) * 0.333333
        c_color = np.minimum(c_color, tC)

        # calculate gradient cost
        tmp = np.ones((m, n)) * tB
        tmp[:, d:n] = fx_r[:, 0:n - d]
        c_grad = np.abs(tmp - fx_l)
        c_grad = np.minimum(c_grad, tG)

        # calculate total cost
        c_tot = al * c_color + (1 - al) * c_grad

        # do the same for the other view
        # SAD
        tmp1 = np.ones((m, n, c)) * tB
        tmp1[:, d:n] = Il_1[:, 0:n - d]
        c1_color = np.abs(tmp1 - Ir_1) * 0.333333
        c1_color = np.sum(c1_color, 2)
        c1_color = np.minimum(c1_color, tC)

        # Grad
        tmp1 = np.ones((m, n)) * tB
        tmp1[:, d:n] = fx_l_1[:, 0:n - d]
        c1_grad = np.abs(tmp1 - fx_r_1)
        c1_grad = np.minimum(c1_grad, tG)

        # total
        c1_tot = al * c1_color + (1 - al) * c1_grad

        # set values
        dispVol[:, :, d] = c_tot
        dispVol1[:, :, d] = c1_tot[:, ::-1]

    print 'Starting Guided image filter. Time taken so far', \
        time.time() - start, 'seconds'

    Il_gf = cv2.ximgproc.createGuidedFilter(Limg, r, eps)
    Ir_gf = cv2.ximgproc.createGuidedFilter(Rimg_1, r, eps)
    q = np.zeros((m, n), dtype=np.float32)
    q1 = np.zeros((m, n), dtype=np.float32)

    # guided image filter
    for d in range(maxDisp):
        p = dispVol[:, :, d].astype(np.float32)
        p1 = dispVol1[:, :, d].astype(np.float32)

        # q = myGIF(Il,p,r,eps)
        Il_gf.filter(p, q)

        p1 = p1[:, ::-1]

        # q1 = myGIF(Il_1, p1, r, eps)
        Ir_gf.filter(p1, q1)

        dispVol[:, :, d] = q
        dispVol1[:, :, d] = q1[:, ::-1]

    print 'Starting minimization. Time taken so far', time.time() - start, \
        'seconds'

    # minimization
    labels_left = np.argmin(dispVol, axis=2)
    labels_right = np.argmin(dispVol1, axis=2)

    # plt.figure()
    # plt.imshow(labels_left,cmap=plt.cm.gray)
    # plt.figure()
    # plt.imshow(labels_right,cmap=plt.cm.gray)

    final_labels = 1 * labels_left

    for y in range(m):
        for x in range(n):
            if np.abs(labels_left[y][x] -
                      labels_right[y][x - labels_left[y][x]]) >= lim:
                final_labels[y][x] = -1

    # fill occluded points - nda
    final_labels_filled = 1 * final_labels
    for y in range(m):
        for x in range(n):
            if final_labels_filled[y][x] <= 0:
                final_labels_filled[y][x] = final_labels_filled[y][x - 1]

    for y in range(m):
        for x in range(maxDisp, 0, -1):
            if final_labels_filled[y][x] <= 0:
                final_labels_filled[y][x] = final_labels_filled[y][x + 1]

    # plt.figure()
    # plt.imshow(final_labels_filled,cmap=plt.cm.gray)

    plt.figure()
    plt.imshow(final_labels, cmap=plt.cm.gray)
    fstr = 'data/res/fcv_norm/' + image + \
        '_fcv_r' + str(r) + '_al' + str(al) + '.jpg'
    # plt.imsave(fstr, final_labels, cmap=plt.cm.gray)
    print "image saved as:" + fstr
    print np.max(final_labels)
    if image == 'mot':
        fstr = 'data/res/fcv_norm/' + image + \
            '_fcv_r' + str(r) + '_al' + str(al) + '.pfm'
        file = open(fstr, 'wb')
        save_pfm(file, final_labels_filled.astype('float32'), scale=1)
        file.close()
    plt.show()
    print 'Script ended. Time taken:', time.time() - start, 'seconds'
