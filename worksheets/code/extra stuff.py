# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:12:34 2016

@author: tt
"""

def sub2ind(array_shape, rows, cols):
    '''
    Found on stackoverview
    '''
    return rows*array_shape[1] + cols

def sub2indWrap(array_shape,rows_arr,cols_arr):
    '''
    wrapper for sub2ind
    '''
    ind = np.zeros(array_shape)

    for y in range(array_shape[0]):
        for x in range(array_shape[1]):
            ind[y][x] = sub2ind(array_shape,y,x)

    return ind.astype(int)