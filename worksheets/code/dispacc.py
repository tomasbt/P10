# disparity accuracy
# z = (b*f)/d ==> d = (b*f)/z
import matplotlib.pyplot as plt
import numpy as np

# scene specs
# - scene depth
dep_max = 1500.0
dep_min = 500.0

# - scene width and height
sw_min = 1000.0
sh_min = 750.0

# sensor specs
# - sensor resolution
s_x = 2592.0
s_y = 1944.0

# - pixel size
px_sz = 0.0022

# - baseline
b = 125.0



z = np.arange(500,1500,2) # 50 cm to 150 cm with 2 mm steps
d = (100.0 * 25.0) / z

print d
