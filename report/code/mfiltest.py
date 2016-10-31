# mean filt test
import numpy as np

num_mat = np.asarray([[1, 2, 3, 5, 6],
                      [2, 4, 6, 1, 3],
                      [3, 5, 6, 10, 6],
                      [7, 4, 4, 8, 9],
                      [1, 4, 3, 2, 1]])

win_r = 1

print 'window size:', win_r * 2 + 1, 'x', win_r * 2 + 1
sum_mat = np.zeros((5, 5), dtype=int)
for y in range(num_mat.shape[0]):
    for x in range(num_mat.shape[1]):
        y0 = y - win_r if y - win_r >= 0 else 0
        y1 = y + win_r + 1 if y + win_r + 1 <= 5 else 5
        x0 = x - win_r if x - win_r >= 0 else 0
        x1 = x + win_r + 1 if x + win_r + 1 <= 5 else 5
        sum_mat[y, x] = np.sum(num_mat[y0:y1, x0:x1])

man_sum_mat = np.asarray([[9, 18, 21, 24, 15],
                          [17, 32, 42, 46, 31],
                          [25, 41, 48, 53, 37],
                          [24, 37, 46, 49, 36],
                          [16, 23, 25, 27, 20]])
# c = (2, 3)
# y0 = c[0] - win_r if c[0] - win_r >= 0 else 0
# y1 = c[0] + win_r + 1 if c[0] + win_r + 1 <= 5 else 5
# x0 = c[1] - win_r if c[1] - win_r >= 0 else 0
# x1 = c[1] + win_r + 1 if c[1] + win_r + 1 <= 5 else 5
# print y0, y1, x0, x1
# print num_mat[y0:y1, x0:x1]
# print np.sum(num_mat[y0:y1, x0:x1])
print man_sum_mat
print sum_mat

ny_sum_p = np.zeros(
    (num_mat.shape[0] + win_r, num_mat.shape[0] + win_r), dtype=int)
for y in range(0, ny_sum_p.shape[0]):
    cum_sum = ny_sum_p[y - 1, 1] if y > 0 else 0
    for x in range(0, ny_sum_p.shape[1]):
        if (y < win_r * 2 + 1) & (x < win_r * 2 + 1):  # Upper Left Corner
            cum_sum = num_mat[y, x] + cum_sum

        elif (y < win_r * 2 + 1) & (x < num_mat.shape[1]):  # Upper border
            cum_sum = num_mat[y, x] - num_mat[y, x - win_r * 2 - 1] + cum_sum

        elif (y < win_r * 2 + 1) & (x >= num_mat.shape[1]):  # Upper right Cor
            cum_sum = - num_mat[y, x - win_r * 2 - 1] + cum_sum

        elif (y < num_mat.shape[0]) & (x < win_r * 2 + 1):  # left border
            cum_sum = num_mat[y, x] - num_mat[y - win_r * 2 - 1, x] + cum_sum

        elif (y < num_mat.shape[0]) & (x >= num_mat.shape[1]):  # right border
            cum_sum = \
                - num_mat[y, x - win_r * 2 - 1] \
                + cum_sum
        elif (y >= num_mat.shape[0]) & (x < win_r * 2 + 1):  # lower left corn
            cum_sum = - num_mat[y - win_r * 2 - 1, x] \
                + cum_sum
        elif (y >= num_mat.shape[0]) & (x < num_mat.shape[1]):  # lower border
            cum_sum = - num_mat[y - win_r * 2 - 1, x] \
                + cum_sum
        elif (y >= num_mat.shape[0]) & (x >= num_mat.shape[1]):
            continue
        else:
            cum_sum = num_mat[y, x] \
                - num_mat[y, x - win_r * 2 - 1] \
                - num_mat[y - win_r * 2 - 1, x] \
                + cum_sum

        ny_sum_p[y, x] = cum_sum

a_sum = ny_sum_p[1:, 1:]
print a_sum
