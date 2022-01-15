#!/usr/bin/env python 
# coding:utf-8

import os
import numpy as np

mock_file = "/Users/mahaixia/VIPERS_MOCK/VIPERS_MOCK_W4_0.5-0.51.txt"
halo_id_file = "/Users/mahaixia/VIPERS_MOCK/bin_halo_number_0.5-0.51.txt"
# mock_data = np.array(pd.read_table(file_name, header=None, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], sep='\s+'))
#%%
with open(halo_id_file, "r") as f:
    for line in f.readlines():
        halo_number = line.strip('\n')
        cmd = "python2.7 cluster.py -i /Users/mahaixia/VIPERS_MOCK/VIPERS_MOCK_W4_0.5-0.51.txt -c 1 2 3 4 --id " \
              + halo_number + " --id_col 11 -s 30.0 >>a.out"
        os.system(cmd)
        print(halo_number)
print("done!")