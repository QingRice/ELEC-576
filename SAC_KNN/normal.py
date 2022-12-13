import sys
import numpy as np
import numpy.matlib
import time
import math
import os
import matplotlib
from itertools import combinations
import matplotlib.pyplot as plt
from mimo_sim_ul import *
import h5py


H_file  = h5py.File('./benchmark/16_4_16_mobile.hdf5', 'r')
H_r = H_file.get('H_r')
H_r = np.transpose(H_r,(2,1,0))
print("H_r shape is:", H_r.shape)

H_i = H_file.get('H_i')
H_i = np.transpose(H_i,(2,1,0))
print("H_r shape is:",H_i.shape)

H = np.squeeze(np.array(H_r + 1j*H_i))
print("H shape is:",H.shape)


N_UE = 1

############# Normalization ##################
MOD_ORDER = np.array([64]) ## 

se_max_ur = np.zeros((400,16)) 

for n_tti in range (0,400):
    H_T = H[n_tti,:,:]
    print('The number of round:',n_tti)
    # print(H_T.shape)
    for n_ur in range (0,16):
        # print(H_T[:,n_ur].shape)
        H_matrix = np.reshape(H_T[:,n_ur],(16,1))
        se_max_ur[n_tti,n_ur], _,_ = data_process(H_matrix,N_UE,MOD_ORDER)

normal_ur = np.zeros(400)
se_max_ur = np.sort(se_max_ur,axis = -1)
for n_tti in range (0,400):
    normal_ur[n_tti] = se_max_ur[n_tti,12] + se_max_ur[n_tti,13] + se_max_ur[n_tti,14] + se_max_ur[n_tti,15]


with h5py.File("./16_4_16_mobile_normal.hdf5", "w") as data_file:
    # data_file.create_dataset("se_max", data=se_max_ur)
    # data_file.create_dataset("H_r", data=H_r)
    # data_file.create_dataset("H_i", data=H_i)
    data_file.create_dataset("normal", data=normal_ur)


############ State SE calculation ##################
# MOD_ORDER = np.array([16]) ## Assume QPSK for all UEs

# se_max_ur = np.zeros((400,16)) 

# for n_tti in range (0,400):
#     H_T = H[n_tti,:,:]
#     print('The number of round:',n_tti)
#     # print(H_T.shape)
#     for n_ur in range (0,16):
#         # print(H_T[:,n_ur].shape)
#         H_matrix = np.reshape(H_T[:,n_ur],(16,1))
#         se_max_ur[n_tti,n_ur], _,_ = data_process(H_matrix,N_UE,MOD_ORDER)

# with h5py.File("./16_4_16_mobile_all.hdf5", "w") as data_file:
#     data_file.create_dataset("se_max", data=se_max_ur)
#     data_file.create_dataset("H_r", data=H_r)
#     data_file.create_dataset("H_i", data=H_i)


# print('Max SE calculation finished and H is stored (400,8,4)\n')



############# Pre-processing calculation ##################
# P_N_UE = 4
# N_UE = 16
# max_episode_length = 400
# user_set = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# actions_num = 0 # number of actions
# for i in range (1,P_N_UE+1):
#     actions_num += len(list(combinations(user_set,i)))

# MOD_ORDER = np.array([16]) ## Assume QPSK for all UEs

# pre_data = np.zeros((400,actions_num,6))

# def sel_ue(action):
#     sum_before = 0
#     for i in range (1,(P_N_UE+1)):
#         sum_before += len(list(combinations(user_set, i)))
#         if ((action+1)>sum_before):
#             continue
#         else:
#             idx = i
#             sum_before -= len(list(combinations(user_set, i)))
#             ue_select = list(combinations(user_set, i))[action-sum_before]
#             break
#     # print('action and sel is:',action, ue_select)
#     return ue_select,idx


# # se_max_ur = np.zeros((400,16)) 

# for n_tti in range (0,400):
#     H_T = H[n_tti,:,:]
#     print('The number of round:',n_tti)
#     # print(H_T.shape)
#     for n_ur in range (0,actions_num):
#         # print(H_T[:,n_ur].shape)
#         ue_select, idx = sel_ue(n_ur)
#         mod_select = np.ones((idx,)) *16 # 16-QAM
#         ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H_T[:,ue_select],(16,-1)),idx,mod_select)
#         pre_data[n_tti,n_ur,0] = ur_se_total
#         pre_data[n_tti,n_ur,1] = ur_min_snr
#         pre_data[n_tti,n_ur,2:2+idx] = ur_se


# with h5py.File("./pre_processing_16_4_16.hdf5", "w") as data_file:
#     data_file.create_dataset("pre", data=pre_data)


# print('Pre-processing finished and stored\n')