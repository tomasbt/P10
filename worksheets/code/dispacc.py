# disparity accuracy
# z = (b*f)/d ==> d = (b*f)/z
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams.update({'figure.autolayout': True})
# scene specs
# - scene depth
dep_max = 1500.0
dep_min = 500.0
#
# # - scene width and height
# sw_min = 1000.0
# sh_min = 750.0
#

# sensor specs
# - sensor resolution
s_x = 2592.0
s_y = 1944.0

# - pixel size
px_sz = 0.0022

# - baseline
b = 125.0
#
#
#
# z = np.arange(500,1500,2) # 50 cm to 150 cm with 2 mm steps

# se sammenhaengen mellem f og scene stoerrelse
f = np.asarray([6,9,16,25,50])

scene_width = ((px_sz*s_x)/f)*dep_max

f2 = np.arange(1,56,0.1)

scene_width2 = ((px_sz*s_x)/f2)*dep_max

plt.figure(figsize=(8,2))
plt.plot(f2,scene_width2,f,scene_width,'ro')
plt.xlabel("Focal length [mm]")
plt.ylabel("Scene width [mm]")
plt.axis([0,55,0,scene_width[0]*1.1])
plt.grid(True)
fstr = '../../report/figures/focalSceWid.jpg'
plt.savefig(fstr)

# disparity precision at 1500 mm if f is changing and therefore scene width
d = np.floor((b*f)/(px_sz*dep_max))
d2 = np.floor((b*f2)/(px_sz*dep_max))
dmin = np.ceil((b*f)/(px_sz*dep_min))
dmin2 = np.ceil((b*f2)/(px_sz*dep_min))
drange = dmin-d
drange2 = dmin2-d2

dispPre = (b*f)/(px_sz*(d))-(b*f)/(px_sz*(d+1))
dispPre2 = (b*f2)/(px_sz*(d2))-(b*f2)/(px_sz*(d2+1))

plt.figure(figsize=(8,2.25))
plt.plot(f2,dispPre2,f,dispPre,'ro')
plt.xlabel("Focal length [mm]")
plt.ylabel("Disparity precision [mm]")
plt.plot([0,55],[2,2],'r-',[0,55],[4,4],'g-')
plt.axis([0,55,0,dispPre[0]*1.1])
plt.grid(True)
fstr2 = '../../report/figures/dispPreFocal.jpg'
plt.savefig(fstr2)

fig = plt.figure(figsize=(8,2.25))
plt.plot(f2,drange2,label='disparity range')
plt.plot(f,drange,'ro')
plt.plot(f2,d2,label='min disparity')
plt.plot(f2,dmin2,label='max disparity')
plt.xlabel("Focal length [mm]")
plt.ylabel("Disparity value [.]")
plt.axis([0,55,0,drange[-1]*1.1])
ax = plt.gca()
# x, y, w, h
pos1 = ax.get_position()
ax.set_position([0.125,0.1,1.05,0.9])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
fstr2 = '../../report/figures/drange.jpg'
plt.savefig(fstr2)

f_2mm = - (px_sz*(2-dep_max)*dep_max)/(2*b)
f_4mm = - (px_sz*(4-dep_max)*dep_max)/(4*b)
sw_2mm = ((px_sz*s_x)/f_2mm)*dep_max
sw_4mm = ((px_sz*s_x)/f_4mm)*dep_max

sw_10mm = ((px_sz*s_x)/10)*dep_max
d_10mm = np.floor((b*10)/(px_sz*dep_max))
dispPre_10 = (b*10)/(px_sz*(d_10mm))-(b*10)/(px_sz*(d_10mm+1))
dmin_10mm = np.ceil((b*10)/(px_sz*dep_min))
drange_10mm = dmin_10mm-d_10mm

sw_20mm = ((px_sz*s_x)/20)*dep_max
d_20mm = np.floor((b*20)/(px_sz*dep_max))
dispPre_20 = (b*20)/(px_sz*(d_20mm))-(b*20)/(px_sz*(d_20mm+1))
dmin_20mm = np.ceil((b*20)/(px_sz*dep_min))
drange_20mm = dmin_20mm-d_20mm

print "2mm pre: f:",f_2mm,"sw",sw_2mm,"4mm pre: f:",f_4mm, "sw:",sw_4mm
print "10f:", sw_10mm, drange_10mm, dispPre_10, '20f:', sw_20mm, drange_20mm, dispPre_20
# plt.show()
