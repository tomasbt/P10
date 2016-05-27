# save and load npy test

import numpy as np
# import h5py

if __name__ == '__main__' or True:
  a = np.random.randint(1,5,(1000,1000,200))
  fname = 'testhdf5file1.npy'
  fname1 = 'testhdf5file.npy'
  # file = h5py.File(fname,'w')
  # file.create_dataset("dset",(4, 6), h5py.h5t.STD_I32BE)

  # test numpy.save

  np.save(fname,a)

  b = np.load(fname)
