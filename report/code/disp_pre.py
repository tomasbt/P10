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
        if -4.98 > zprime[i] > -5.02:
            print i
            disp = i
        elif -9.9 > zprime[i] > -10.1:
            print i
            disp_10mm = i
        if zprime[i] == 5:
            print 'argh', i

    z = baseline * focalLength / (disp * pixelSize[0])
    print z
    z = baseline * focalLength / (disp_10mm * pixelSize[0])
    print z

    plt.plot(zprime)
    plt.axis([0, 3000, -10, 5])
    plt.show()
