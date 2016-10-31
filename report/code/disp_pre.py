'''
This script is for calculating the requirements for new values
'''

import numpy as np
from matplotlib import pyplot as plt


if __name__ == 'main' or True:
    '''
    requirements:
    distance 0.5 - 1.5 m --> 500 mm - 1500 mm
    scene size at max dist 1.5 x 1.5 m --> 1500 mm x 1500 mm
    precision in dist range =< 5 mm
    base line =< 100 mm
    sensor = IMX264
    res 2464 x 2056 pixels
    pixel size 3.45 um --> 0.00345 mm
    '''
    # known specs and requirements:
    # Sensor specs:
    resolution = (2464, 2056)  # h,v
    pixelSize = (0.00345, 0.00345)  # (h,v) in mm

    # distance range:
    distRange = (500, 1500)  # (min, max) in mm

    # baseline: might change
    baseline = 100  # max 10 cm

    # scene size
    sceneSize = (1500, 1500)   # (h,v) in mm

    # calculate the focal length
    fH = distRange[1] * pixelSize[0] * resolution[0] / sceneSize[0]
    fV = distRange[1] * pixelSize[1] * resolution[1] / sceneSize[1]
    focalLength = np.min([fH, fV])

    print 'Horizontal focal length', fH, 'Vertical focal length', fV, \
        'Chosen focal length', focalLength

    # Create an array for d. This will be used as disparity values and will
    #   have increments of 1 since the disparity is discrete.
    d = np.arange(1, 3000)

    # zprime is the differentiation, dz/dd, where z = b*f/(pixelSize*d)
    zprime = -(baseline * focalLength) / (pixelSize[0] * (d**2))

    # Calculate the disparity precision is around 5 mm:
    for i in range(len(zprime)):
        if zprime[i] >= -5:
            print i
            disp = i + 1
            break

    for i in range(0, disp):
        if zprime[i] >= -10:
            print i
            disp10 = i + 1
            break

    z = baseline * focalLength / (disp * pixelSize[0])
    print z
    z = baseline * focalLength / (disp10 * pixelSize[0])
    print z

    z = baseline * focalLength / (d * pixelSize[0])

    disp_500 = np.ceil((baseline * focalLength) /
                       (distRange[0] * pixelSize[0]))
    disp_1500 = np.floor((baseline * focalLength) /
                         (distRange[1] * pixelSize[0]))
    disp_range = disp_500 - disp_1500
    print disp_500, disp_1500, disp_range

    # plt.figure(figsize=(8, 2.25))
    # plt.plot(d, zprime)
    # plt.axis([137, 412, -10, 0])
    #
    # plt.figure(figsize=(8, 2.25))
    # plt.plot(d, z)
    # plt.axis([137, 412, 500, 1500])

    fig = plt.figure(figsize=(8, 2.5))
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.plot(z, zprime)
    plt.xlabel("Distance [mm]")
    plt.ylabel("Disparity precision [mm]")
    plt.xticks(np.arange(500, 1501, 100))
    plt.axis([500, 1500, -10, 0])
    plt.grid(True)
    fstr = '../figures/dpre.jpg'
    plt.savefig(fstr)

    plt.show()
