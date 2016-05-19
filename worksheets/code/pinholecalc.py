## pinhole calculator
import numpy as np
import matplotlib.pyplot as plt

def normCalc(Zmax,Zmin,rx,ry,ps,sx,sy,b):
    """
    You specify the specification and the system will calculate all the results
    """
    # calculate sensor size
    wx = rx*ps
    wy = ry*ps

    # calculate focal length
    fx = 1.0 * Zmax * wx / sx
    fy = 1.0 * Zmax * wy / sy
    fo = np.min((fx,fy))

    # calculate disparity
    d_min = (1.0*fo*b)/Zmax
    d_max = (1.0*fo*b)/Zmin
    dP_min = d_min/ps
    dP_max = d_max/ps
    P_range = dP_max - dP_min

    # resolution
    dZ = Zmax - (1.0*fo*b/(d_min+ps))
    dX = (Zmax*wx/fo)/rx
    dY = (Zmax*wy/fo)/ry

    return [fo, d_min, d_max, dP_min, dP_max, P_range, dZ, dX, dY]

if __name__ == 'main' or True:
    """
    variables:
    rx, ry      : image resolution [px]
    ps          : Pixel size [mm]
    sx, sy      : Scene width & height [mm]
    wx, wy      : Sensor width & height [mm]
    Zmax        : Max scene depth [mm]
    Zmin        : Min scene depth [mm]
    fx, fy, fo  : Focal length in x, y direction and final focal length
    b           : baseline [mm]
    d           : disparity [mm]
    """

    print "\n\n\n\n\n\n"
    print "###############################"
    print "########### NEW RUN ###########"
    print "###############################"

    # scene max og min depth and size
    Zmax = 1500
    Zmin = 500
    sx = 1000
    sy = 500

    # pixel size
    ps = 0.0022

    # baseline
    b = 125

    # resolution
    rx = 1280
    ry = 720

    # what result do I want to compare to:
    #  0   1      2      3       4       5        6   7   8
    # [fo, d_min, d_max, dP_min, dP_max, P_range, dZ, dX, dY]
    for ii in range(9):
        resChe = ii
        resStrArr = ["fo [mm]","d_min [mm]","d_max [mm]","dP_min [px]","dP_max [px]","P_range [px]","dZ [mm]","dX [mm]","dY [mm]"]
        resStr = resStrArr[ii]

        print "Looking at",resStr,"now --------------------------------------------------------------------"
        # check baseline influence on dZ
        b_vals = np.arange(50,201,10)
        b_plot = np.zeros((2,len(b_vals)))
        for i in range(len(b_vals)):
            res = normCalc(Zmax, Zmin, rx, ry, ps, sx, sy, b_vals[i])
            b_plot[0][i] = res[resChe]
            b_plot[1][i] = b_vals[i]
        plt.figure()
        plt.title('baseline influence')
        plt.xlabel('Baseline [mm]')
        plt.ylabel(resStr)
        # plt.axis([50,150,b_plot[0].min(),b_plot[0].max()])
        plt.grid(True)
        plt.plot(b_plot[1,:],b_plot[0,:])
        plt.show()



        # check resolution influence on dZ
        rx_vals = [640,741,800,1024,1280,1366,1600,1920,2048,2592]
        ry_vals = [480,497,600,768 ,800 ,768 ,1200,1080,1536,1944]
        rx_plot = np.zeros((3,len(rx_vals)))
        print "tested resolutions:"
        for i in range(len(rx_vals)):
            res = normCalc(Zmax, Zmin, rx_vals[i], ry_vals[i], ps, sx, sy, b)
            rx_plot[0][i] = rx_vals[i]
            rx_plot[1][i] = ry_vals[i]
            rx_plot[2][i] = res[resChe]
            print "re[",i,"]:",rx_vals[i],"x",ry_vals[i]
        plt.figure()
        plt.title('resolution influence')
        plt.xlabel('x resolution [pixel]')
        plt.ylabel(resStr)
        # plt.axis([,,b_plot[0].min(),b_plot[0].max()])
        plt.grid(True)
        plt.plot(rx_plot[0,:],rx_plot[2,:])
        plt.show()

        # check resolution aspect ratio influence on dZ
        ry_ratio = np.arange(0.25,1.26,0.25)
        ry_plot = np.zeros((2,len(sy_ratio)))
        for i in range(len(ry_ratio)):
            res = normCalc(Zmax, Zmin, rx, rx*ry_ratio[i], ps, sx, sy, b)
            ry_plot[0][i] = ry_ratio[i]
            ry_plot[1][i] = res[resChe]
        plt.figure()
        plt.title('resolution aspect ratio influence (width = 800 px)(scene aspect ratio = 0.5)')
        plt.xlabel('resolution aspect ratio [.]')
        plt.ylabel(resStr)
        # plt.text(0.55,20,'when the aspect ratio is lower\nthan scene aspect ratio\ndZ rises')
        # plt.axis([,,b_plot[0].min(),b_plot[0].max()])
        plt.grid(True)
        plt.plot(ry_plot[0,:],ry_plot[1,:])
        plt.show()

        # check scene size influence on dZ
        sx_vals = np.arange(500,2001,250)
        sx_plot = np.zeros((2,len(sx_vals)))
        for i in range(len(sx_vals)):
            res = normCalc(Zmax, Zmin, rx, ry, ps, sx_vals[i], sx_vals[i]*0.75, b)
            sx_plot[0][i] = sx_vals[i]
            sx_plot[1][i] = res[resChe]
        plt.figure()
        plt.title('scene size influence (aspect ratio = 4/3)')
        plt.xlabel('scene width resolution [mm]')
        plt.ylabel(resStr)
        # plt.axis([,,b_plot[0].min(),b_plot[0].max()])
        plt.grid(True)
        plt.plot(sx_plot[0,:],sx_plot[1,:])
        plt.show()

        # check scene aspect ratio influence on dZ
        sy_ratio = np.arange(0.25,1.26,0.25)
        sy_plot = np.zeros((2,len(sy_ratio)))
        for i in range(len(sy_ratio)):
            res = normCalc(Zmax, Zmin, rx, rx*0.75, ps, 1000, 1000*sy_ratio[i], b)
            sy_plot[0][i] = sy_ratio[i]
            sy_plot[1][i] = res[resChe]
        plt.figure()
        plt.title('scene aspect ratio influence (width = 1000 mm)(resolution aspect ratio = 0.75)')
        plt.xlabel('scene aspect ratio [.]')
        plt.ylabel(resStr)
        # plt.text(0.835,18.5,'when the aspect ratio is larger\nthan resolution aspect ratio\ndZ rises')
        # plt.axis([,,b_plot[0].min(),b_plot[0].max()])
        plt.grid(True)
        plt.plot(sy_plot[0,:],sy_plot[1,:])
        plt.show()

        # check Zmin influence on dZ
        Zmin_vals = np.arange(250,1251,250)
        Zmin_plot = np.zeros((2,len(Zmin_vals)))
        for i in range(len(Zmin_vals)):
            res = normCalc(Zmax, Zmin_vals[i], rx, rx*0.75, ps, sx, sx*0.75, b)
            Zmin_plot[0][i] = Zmin_vals[i]
            Zmin_plot[1][i] = res[resChe]
        plt.figure()
        plt.title('Zmin influence')
        plt.xlabel('Zmin [mm]')
        plt.ylabel(resStr)
        # plt.axis([,,b_plot[0].min(),b_plot[0].max()])
        plt.grid(True)
        plt.plot(Zmin_plot[0,:],Zmin_plot[1,:])
        plt.show()

        # check Zmax influence on dZ
        Zmax_vals = np.arange(750,2251,250)
        Zmax_plot = np.zeros((2,len(Zmax_vals)))
        for i in range(len(Zmax_vals)):
            res = normCalc(Zmax_vals[i], Zmin, rx, rx*0.75, ps, sx, sx*0.75, b)
            Zmax_plot[0][i] = Zmax_vals[i]
            Zmax_plot[1][i] = res[resChe]
        plt.figure()
        plt.title('Zmax influence')
        plt.xlabel('Zmax [mm]')
        plt.ylabel(resStr)
        # plt.axis([,,b_plot[0].min(),b_plot[0].max()])
        plt.grid(True)
        plt.plot(Zmax_plot[0,:],Zmax_plot[1,:])
        plt.show()

        # check Pixel size influence on dZ
        ps_vals = np.arange(0.0010,0.0041,0.0002)
        ps_plot = np.zeros((2,len(ps_vals)))
        for i in range(len(ps_vals)):
            res = normCalc(Zmax, Zmin, rx, rx*0.75, ps_vals[i], sx, sx*0.75, b)
            ps_plot[0][i] = ps_vals[i]
            ps_plot[1][i] = res[0]
        plt.figure()
        plt.title('ps influence')
        plt.xlabel('ps [mm]')
        plt.ylabel(resStr)
        # plt.axis([,,b_plot[0].min(),b_plot[0].max()])
        plt.grid(True)
        plt.plot(ps_plot[0,:],ps_plot[1,:])
        plt.show()
