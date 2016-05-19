## extra code snippets

# # Read pgm image file -- grayscale ---------------------------------------------
# def read_pgm(filename, byteorder='>'):
#     """Return image data from a raw PGM file as numpy array.
#
#     Format specification: http://netpbm.sourceforge.net/doc/pgm.html
#
#     Gain from STACK OVERFLOW
#     """
#     with open(filename, 'rb') as f:
#         buffer = f.read()
#     try:
#         header, width, height, maxval = re.search(
#             b"(^P5\s(?:\s*#.*[\r\n])*"
#             b"(\d+)\s(?:\s*#.*[\r\n])*"
#             b"(\d+)\s(?:\s*#.*[\r\n])*"
#             b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
#     except AttributeError:
#         raise ValueError("Not a raw PGM file: '%s'" % filename)
#     return np.frombuffer(buffer,
#                             dtype='int', #'u1' if int(maxval) < 256 else byteorder+'u2',
#                             count=int(width)*int(height),
#                             offset=len(header)
#                             ).reshape((int(height), int(width)))


# # SWS update rule for horizontal directions ------------------------------------
# def sws_update_horr(c_d,rimg,x,y,f_data,direct='r'):
#     '''
#     Update rule for successive weighted sum
#
#     limg   - data from the left image
#     rimg   - data from the right image
#     x, y   - coordinates for the current pixel
#     direct - which scan order is going to be preformed. either 'r' or 'l'
#     f_data - former data
#     alpha  - parameter for cost calculation
#     d      - Disparity
#
#     not used anymore !!!!!
#     '''
#
#     if direct == 'r':
#         # if the direction is right then do the following
#
#         curr_c = c_d + permeability(rimg,x,y,direct='r')*f_data
#
#     elif direct == 'l':
#         # if the direction is left then do the following
#
#         curr_c = c_d + permeability(rimg,x,y,direct='l')*f_data
#
#     # print 'x,y:',x,y,'and d:',d,'and cost:',curr_c
#     return curr_c
#
# # SWS update rule for vertical direction ---------------------------------------
# def sws_update_vert(img,horr_data,x,y,f_data,d,direct='t'):
#     '''
#     Update rule for successive weighted sum
#
#     img - image data for permeability calculation
#     x,y - coordinates
#     f_data - former data
#     d - disparity
#     direct - Direction. either 't' or 'b'
#     '''
#     if direct == 't':
#         # if the direction is towards top then do the following
#
#         curr_c = horr_data + permeability(img,x,y,direct='t')*f_data
#
#     elif direct == 'b':
#         # if the direction is towards bottom then do the following
#
#         curr_c = horr_data + permeability(img,x,y,direct='b')*f_data
#
#     return curr_c
