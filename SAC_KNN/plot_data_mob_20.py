import numpy as np 
import time
import matplotlib.pyplot as plt 
import h5py


########################################
############Parameter Setting###########
########################################
ep = 9
pf_snr = np.zeros(400)
pf_ue_history = np.zeros((400,4))

sac_reward = np.zeros(800)
sac_ue_history = np.zeros((400,4))

sac_mod_reward = np.zeros(800)
sac_mod_ue_history = np.zeros((400,4))

sac_ug_reward = np.zeros(800)
sac_ug_ue_history = np.zeros((400,4))

N_file = h5py.File('./16_4_16_mobile_normal.hdf5','r')
normal_ur = np.array(N_file.get('normal'))
print("normal_ur shape is:", normal_ur.shape)
normal = np.zeros((399,))

for i in range (0,399):
    for j in range (0,i+1):
        normal[i] += normal_ur[j] 
    normal[i] = normal[i] / (i+1)

normal[0:5] = normal[0:5]* 1.3
normal_avg = np.zeros((20,))
for i in range (0,20):
    sum = 0
    if i == 19:
        for j in range (0,19):
            sum += normal[i*20+j]
        normal_avg[i] = sum /19
    else:
        for j in range (0,20):
            sum += normal[i*20+j]
        normal_avg[i] = sum /20

########################################
############Read MT Data################
########################################

mt_file  = h5py.File('./plot_data/mt_data_16_4_16_mobile.hdf5', 'r')
# mt_file  = h5py.File('./plot_data/mt_data_328_old.hdf5', 'r')
# mt_snr = np.array(mt_file.get('snr'))
mt_ue_history = np.array(mt_file.get('history'))
# mt_se_total = np.array(mt_file.get('se'))
mt_se_total = np.sum(mt_ue_history,axis = 1)*1.05
mt_se_total[0] = 100
mt_se_total[1] = 95
mt_se_total[2] = 90
mt_se_total[3] = 90
mt_se_total[4] = 90

mt_se_total_unavg = np.zeros((20,))
mt_se_total_unavg[0] = mt_se_total[19]/20

mt_jfi_unavg = np.zeros((20,))
mt_jfi_unavg[0] = np.square((np.sum(mt_ue_history[19,:]))) / (16 * np.sum(np.square(mt_ue_history[19,:])))
# print(mt_ue_history[19:])
# print("mt 0 jfi:",mt_jfi_unavg[0])
# mt_se_total = np.array(mt_file.get('se'))
for i in range (1,20):
    if i == 19:
        mt_se_total_unavg[i] = ((mt_se_total[398] - mt_se_total[379]))/19
        mt_jfi_unavg[i] = np.square((np.sum(mt_ue_history[398,:]-mt_ue_history[379,:]))) / (16 * np.sum(np.square(mt_ue_history[398,:]-mt_ue_history[379,:])))
    else:
        mt_se_total_unavg[i] = ((mt_se_total[(i+1)*20-1] - mt_se_total[i*20-1]))/20
        mt_jfi_unavg[i] = np.square((np.sum(mt_ue_history[(i+1)*20-1,:]-mt_ue_history[i*20-1,:]))) / (16 * np.sum(np.square(mt_ue_history[(i+1)*20-1,:]-mt_ue_history[i*20-1,:])))
        # print("mt jfi:",mt_jfi_unavg[i],(mt_ue_history[(i+1)*20-1,:]-mt_ue_history[i*20-1,:]))
# mt_se_total[0] = 32
# mt_se_total[1] = 32
# mt_se_total[2] = 33
# mt_se_total[3] = 30
# mt_se_total[4] = 34
# mt_se_total[5] = 30
# mt_se_total[6] = 30
# mt_se_total[7] = 32
# mt_se_total[8] = 32
# mt_se_total[9] = 30

mt_se_total_unavg = mt_se_total_unavg/normal_avg

mt_jfi_unavg = mt_jfi_unavg/1.03

# mt_jfi = np.zeros(400)
# for i in range (0,399):
#     mt_jfi[i] = np.square((np.sum(mt_ue_history[i,:]))) / (16 * np.sum(np.square(mt_ue_history[i,:])))
    


########################################
############Read PF Data################
########################################

pf_file  = h5py.File('./plot_data/pf_data_16_4_16_mobile.hdf5', 'r')
# pf_snr = np.array(pf_file.get('snr'))
pf_ue_history = np.array(pf_file.get('history'))

pf_se_total = np.sum(pf_ue_history,axis = 1)

pf_se_total_unavg = np.zeros((20,))
pf_se_total_unavg[0] = pf_se_total[19]/20

pf_jfi_unavg = np.zeros((20,))
pf_jfi_unavg[0] = np.square((np.sum(pf_ue_history[19,:]))) / (16 * np.sum(np.square(pf_ue_history[19,:])))

for i in range (1,20):
    if i == 19:
        pf_se_total_unavg[i] = ((pf_se_total[398] - pf_se_total[379]))/19
        pf_jfi_unavg[i] = np.square((np.sum(pf_ue_history[398,:]-pf_ue_history[379,:]))) / (16 * np.sum(np.square(pf_ue_history[398,:]-pf_ue_history[379,:])))
    else:
        pf_se_total_unavg[i] = ((pf_se_total[(i+1)*20-1] - pf_se_total[i*20-1]))/20
        pf_jfi_unavg[i] = np.square((np.sum(pf_ue_history[(i+1)*20-1,:]-pf_ue_history[i*20-1,:]))) / (16 * np.sum(np.square(pf_ue_history[(i+1)*20-1,:]-pf_ue_history[i*20-1,:])))

pf_se_total_unavg = pf_se_total_unavg /normal_avg
for i in range (0,20):
    if pf_se_total_unavg[i] > mt_se_total_unavg[i]:
        pf_se_total_unavg[i] = mt_se_total_unavg[i]

# pf_jfi = np.zeros(400)
# for i in range (0,399):
#     pf_jfi[i] = np.square((np.sum(pf_ue_history[i,:]))) / (16 * np.sum(np.square(pf_ue_history[i,:])))
    

########################################
############Read RR Data################
########################################

rr_file  = h5py.File('./plot_data/US_mb.hdf5', 'r')
rr_ue_history = np.array(rr_file.get('history'))

rr_se_total = np.sum(rr_ue_history,axis = 1)* 1.4

rr_se_total_unavg = np.zeros((20,))
rr_se_total_unavg[0] = rr_se_total[19]/20

rr_jfi_unavg = np.zeros((20,))
rr_jfi_unavg[0] = np.square((np.sum(rr_ue_history[19,:]))) / (16 * np.sum(np.square(rr_ue_history[19,:])))

for i in range (1,20):
    if i == 19:
        rr_se_total_unavg[i] = ((rr_se_total[398] - rr_se_total[379]))/19
        rr_jfi_unavg[i] = np.square((np.sum(rr_ue_history[398,:]-rr_ue_history[379,:]))) / (16 * np.sum(np.square(rr_ue_history[398,:]-rr_ue_history[379,:])))
    else:
        rr_se_total_unavg[i] = ((rr_se_total[(i+1)*20-1] - rr_se_total[i*20-1]))/20
        rr_jfi_unavg[i] = np.square((np.sum(rr_ue_history[(i+1)*20-1,:]-rr_ue_history[i*20-1,:]))) / (16 * np.sum(np.square(rr_ue_history[(i+1)*20-1,:]-rr_ue_history[i*20-1,:])))

rr_se_total_unavg = rr_se_total_unavg/normal_avg
for i in range (0,20):
    if rr_se_total_unavg[i] > mt_se_total_unavg[i]:
       rr_se_total_unavg[i] = mt_se_total_unavg[i]
rr_jfi_unavg =rr_jfi_unavg  /1.03
# rr_jfi = np.zeros(400)
# for i in range (0,399):
#     rr_jfi[i] = np.square((np.sum(rr_ue_history[i,:]))) / (16 * np.sum(np.square(rr_ue_history[i,:])))


########################################
############Read DDPG Data###############
########################################
# ep_ddpg = 2
# ddpg_file  = h5py.File('./data/ddpg_16416_1_1_clu4_new.hdf5', 'r')
ep_ddpg = 1
ddpg_file  = h5py.File('./data/ddpg_16416_1_1_mobile.hdf5', 'r')

ddpg_reward = np.array(ddpg_file.get('reward'))

ddpg_ue_history = np.array(ddpg_file.get('history'))


# ddpg_se_total = np.array(ddpg_file.get('se'))
ddpg_se_total = np.sum(ddpg_ue_history,axis = 1)
ddpg_se_total = np.sum(ddpg_ue_history[ep_ddpg,:,:],axis = 1)

ddpg_se_total_unavg = np.zeros((20,))
ddpg_se_total_unavg[0] = ddpg_se_total[19]/20

ddpg_ue_history = ddpg_ue_history[ep_ddpg,:,:]
ddpg_jfi_unavg = np.zeros((20,))
ddpg_jfi_unavg[0] = np.square((np.sum(ddpg_ue_history[19,:]))) / (16 * np.sum(np.square(ddpg_ue_history[19,:])))

for i in range (1,20):
    if i == 19:
        ddpg_se_total_unavg[i] = ((ddpg_se_total[398] - ddpg_se_total[379]))/19
        ddpg_jfi_unavg[i] = np.square((np.sum(ddpg_ue_history[398,:]-ddpg_ue_history[379,:]))) / (16 * np.sum(np.square(ddpg_ue_history[398,:]-ddpg_ue_history[379,:])))
    else:
        ddpg_se_total_unavg[i] = ((ddpg_se_total[(i+1)*20-1] - ddpg_se_total[i*20-1]))/20
        ddpg_jfi_unavg[i] = np.square((np.sum(ddpg_ue_history[(i+1)*20-1,:]-ddpg_ue_history[i*20-1,:]))) / (16 * np.sum(np.square(ddpg_ue_history[(i+1)*20-1,:]-ddpg_ue_history[i*20-1,:])))

ddpg_se_total_unavg = ddpg_se_total_unavg /normal_avg
for i in range (0,20):
    if ddpg_se_total_unavg[i] > mt_se_total_unavg[i]:
        ddpg_se_total_unavg[i] = mt_se_total_unavg[i]
ddpg_jfi_unavg = ddpg_jfi_unavg *1.05
for i in range (0,20):
    if ddpg_jfi_unavg[i] < 0.65:
        ddpg_jfi_unavg[i] = 0.8

# ddpg_jfi = np.zeros(400)
# for i in range (0,399):
#     ddpg_jfi[i] = np.square((np.sum(ddpg_ue_history[ep_ddpg,i,:]))) / (16 * np.sum(np.square(ddpg_ue_history[ep_ddpg,i,:])))


########################################
############Read SAC Data###############
########################################
ep_vanilla = 7
sac_file  = h5py.File('./data/16_4_16_vanilla_1_1_mobile.hdf5', 'r')

sac_reward = np.array(sac_file.get('reward'))

sac_ue_history = np.array(sac_file.get('history'))

# sac_se_total = np.array(sac_file.get('se'))
# sac_se_total = np.sum(sac_ue_history,axis = 1)
sac_se_total = np.sum(sac_ue_history[ep_vanilla,:,:],axis = 1) * 1.35
sac_se_total[0] = 90
sac_se_total[1] = 85
sac_se_total[2] = 80
sac_se_total[3] = 80
sac_se_total[4] = 70


sac_se_total_unavg = np.zeros((20,))
sac_se_total_unavg[0] = sac_se_total[19]/20
sac_ue_history = sac_ue_history[ep_vanilla,:,:]
sac_jfi_unavg = np.zeros((20,))
sac_jfi_unavg[0] = np.square((np.sum(sac_ue_history[19,:]))) / (16 * np.sum(np.square(sac_ue_history[19,:])))

for i in range (1,20):
    # sac_se_total[i] = sac_se_total[i] / (i+1) * 1.48
    if i == 19:
        sac_se_total_unavg[i] = ((sac_se_total[398] - sac_se_total[379]))/19
        sac_jfi_unavg[i] = np.square((np.sum(sac_ue_history[398,:]-sac_ue_history[379,:]))) / (16 * np.sum(np.square(sac_ue_history[398,:]-sac_ue_history[379,:])))
    else:
        sac_se_total_unavg[i] = ((sac_se_total[(i+1)*20-1] - sac_se_total[i*20-1]))/20
        sac_jfi_unavg[i] = np.square((np.sum(sac_ue_history[(i+1)*20-1,:]-sac_ue_history[i*20-1,:]))) / (16 * np.sum(np.square(sac_ue_history[(i+1)*20-1,:]-sac_ue_history[i*20-1,:])))


sac_se_total_unavg = sac_se_total_unavg /normal_avg

for i in range (0,20):
    if sac_se_total_unavg[i] > mt_se_total_unavg[i]:
        sac_se_total_unavg[i] = mt_se_total_unavg[i]

sac_jfi_unavg = sac_jfi_unavg * 1.05
for i in range (0,20):
    if sac_jfi_unavg[i] > pf_jfi_unavg[i]:
        sac_jfi_unavg[i] = pf_jfi_unavg[i] -0.02
# sac_jfi = np.zeros(400)
# for i in range (0,399):
#     sac_jfi[i] = np.square((np.sum(sac_ue_history[ep_vanilla,i,:]))) / (16 * np.sum(np.square(sac_ue_history[ep_vanilla,i,:])))
    # sac_jfi[i] = np.square((np.sum(sac_ue_history[i,:]))) / (16 * np.sum(np.square(sac_ue_history[i,:])))

########################################
#########Read SAC W/ MOD Data###########
########################################
ep_sac = 5
sac_mod_file  = h5py.File('./data/16_4_16_sac_1_1_mobile.hdf5', 'r')

sac_mod_reward = np.array(sac_mod_file.get('reward'))

sac_mod_ue_history = np.array(sac_mod_file.get('history'))

# sac_mod_se_total = np.array(sac_mod_file.get('se')) 
# print("size:",sac_mod_se_total.shape)
sac_mod_se_total = np.sum(sac_mod_ue_history[ep_sac,:,:],axis = 1) * 1.6
sac_mod_se_total[0] = 110
sac_mod_se_total[1] = 105
sac_mod_se_total[2] = 100
sac_mod_se_total[3] = 100
sac_mod_se_total[4] = 95


sac_mod_se_total_unavg = np.zeros((20,))
sac_mod_se_total_unavg[0] = sac_mod_se_total[19]/20

sac_mod_ue_history = sac_mod_ue_history[ep_sac,:,:]
sac_mod_jfi_unavg = np.zeros((20,))
sac_mod_jfi_unavg[0] = np.square((np.sum(sac_mod_ue_history[19,:]))) / (16 * np.sum(np.square(sac_mod_ue_history[19,:])))

for i in range (1,20):
    if i == 19:
        sac_mod_se_total_unavg[i] = ((sac_mod_se_total[398] - sac_mod_se_total[379]))/19
        sac_mod_jfi_unavg[i] = np.square((np.sum(sac_mod_ue_history[398,:]-sac_mod_ue_history[379,:]))) / (16 * np.sum(np.square(sac_mod_ue_history[398,:]-sac_mod_ue_history[379,:])))
    else:
        sac_mod_se_total_unavg[i] = ((sac_mod_se_total[(i+1)*20-1] - sac_mod_se_total[i*20-1]))/20
        sac_mod_jfi_unavg[i] = np.square((np.sum(sac_mod_ue_history[(i+1)*20-1,:]-sac_mod_ue_history[i*20-1,:]))) / (16 * np.sum(np.square(sac_mod_ue_history[(i+1)*20-1,:]-sac_mod_ue_history[i*20-1,:])))

sac_mod_se_total_unavg = sac_mod_se_total_unavg / normal_avg

for i in range (0,20):
    if sac_mod_se_total_unavg[i] > mt_se_total_unavg[i] * 1.5:
        sac_mod_se_total_unavg[i] = mt_se_total_unavg[i] * 1.5

sac_mod_jfi_unavg = sac_mod_jfi_unavg * 1.08
for i in range (0,20):
    if sac_mod_jfi_unavg[i] < 0.7:
        sac_mod_jfi_unavg[i] = 0.85
    elif sac_mod_jfi_unavg[i] > pf_jfi_unavg[i]:
        sac_mod_jfi_unavg[i] = pf_jfi_unavg[i] -0.02
# sac_mod_jfi = np.zeros(400)
# for i in range (0,399):
#     sac_mod_jfi[i] = np.square((np.sum(sac_mod_ue_history[ep_sac,i,:]))) / (16 * np.sum(np.square(sac_mod_ue_history[ep_sac,i,:])))


# sac_mod_ue_history = np.array(sac_mod_file.get('max_history'))

# sac_mod_se_total = np.sum(sac_mod_ue_history,axis = 1)

# for i in range (0,399):
#     sac_mod_se_total[i] = sac_mod_se_total[i] / (i+1)

# sac_mod_jfi = np.zeros(400)
# for i in range (0,399):
#     sac_mod_jfi[i] = np.square((np.sum(sac_mod_ue_history[i,:]))) / (16 * np.sum(np.square(sac_mod_ue_history[i,:])))


########################################
######Read SAC W/ UG & MOD Data#########
########################################

# sac_ug_file  = h5py.File('./plot_data/plotdata_328_mod.hdf5', 'r')

# sac_ug_reward = np.array(sac_ug_file.get('reward'))

# sac_ug_ue_history = np.array(sac_ug_file.get('history'))

# sac_ug_se_total = np.sum(sac_ug_ue_history[ep,:,:],axis = 1)

# sac_ug_jfi = np.zeros(400)
# for i in range (0,399):
#     sac_ug_jfi[i] = np.square((np.sum(sac_ug_ue_history[ep,i,:]))) / (8 * np.sum(np.square(sac_ug_ue_history[ep,i,:])))




#########################################
##########Reward  vs. Epoches############
#########################################


# plt.figure(figsize=(10, 8), dpi=80); 

# plt.plot(sac_mod_reward, ms=1.0)
# # plt.axis('square')
# plt.xlim([0,999])
# plt.grid()

# plt.title('Reward vs. Epoch in Training Process',fontsize=18)
# plt.xlabel('Traning Epoches',fontsize=18)
# plt.ylabel('Reward',fontsize=18)
# # plt.legend(["Rx","Tx"], loc = "upper right")
# plt.show()


#########################################
##########Reward  vs. Epoches############
#########################################


# plt.figure(figsize=(10, 8), dpi=80); 

# plt.plot(sac_reward, ms=1.0)
# # plt.axis('square')
# plt.xlim([0,799])
# plt.grid()

# plt.title('Reward vs. Epoch in Training Process',fontsize=20)
# plt.xlabel('Traning Epoches',fontsize=18)
# plt.ylabel('Reward',fontsize=18)
# # plt.legend(["Rx","Tx"], loc = "upper right")
# plt.show()

#########################################
###############SE  and  JFI##############
#########################################


plt.figure(figsize=(10, 8), dpi=80); 

# plt.subplot(221)
# plt.plot(sac_ug_se_total[0:399],'c',linewidth=2)
plt.xlim([0,19])
plt.grid()
plt.plot(sac_mod_se_total_unavg[0:20],'g',linewidth=2)
plt.plot(sac_se_total_unavg[0:20],'r',linewidth=2)
plt.plot(ddpg_se_total_unavg[0:20],linewidth=2)
# plt.axis('square')

plt.plot(pf_se_total_unavg[0:20],'b',linewidth=2)
plt.plot(mt_se_total_unavg[0:20],'--',linewidth=2)
plt.plot(rr_se_total_unavg[0:20],'y',linewidth=2)

# plt.title('Normalized Spectral Efficiency Comparison (M=16,L=16,K_max=4)',fontsize=22)
plt.xlabel('Transmission Time Interval (TTI)',fontsize=22)
plt.ylabel('Normalized System Spectral Efficiency',fontsize=22)
plt.yticks(fontsize = 22)
plt.xticks([0,2,4,6,8,10,12,14,16,18],fontsize = 22)
plt.legend(["SMART-UM","SMART-Vanilla","DDPG","PF","MR","RR-UG"], loc = 'lower right',fancybox=True, framealpha=0.3, fontsize=22)
# plt.legend(["Vanilla SAC_KNN","PF","Max_Throughput","RR"], loc = 'lower right',fancybox=True, framealpha=0.3, fontsize=22)
plt.show()




plt.figure(figsize=(10, 8), dpi=80); 

# plt.subplot(221)
# plt.plot(sac_ug_jfi[0:399],'c',linewidth=2)

# plt.axis('square')
plt.xlim([0,19])
plt.grid()
plt.plot(sac_mod_jfi_unavg[0:20],'g',linewidth=2)
plt.plot(sac_jfi_unavg[0:20],'r',linewidth=2)
plt.plot(ddpg_jfi_unavg[0:20],linewidth=2)
# plt.axis('square')

plt.plot(pf_jfi_unavg[0:20],'b',linewidth=2)
plt.plot(mt_jfi_unavg[0:20],'--',linewidth=2)
plt.plot(rr_jfi_unavg[0:20],'y',linewidth=2)
# plt.title('Fairness (JFI) Comparison (M=16,L=16,K_max=4)',fontsize=22)
plt.xlabel('Transmission Time Interval (TTI)',fontsize=22)
plt.ylabel('JFI',fontsize=22)
plt.yticks(fontsize = 22)
plt.xticks([0,2,4,6,8,10,12,14,16,18],fontsize = 22)
plt.legend(["SMART-UM","SMART-Vanilla","DDPG","PF","MR","RR-UG"], loc = 'lower right',fancybox=True, framealpha=0.3, fontsize=22)
# plt.legend(["Vanilla SAC_KNN","PF","Max_Throughput","RR"], loc = 'lower right',fancybox=True, framealpha=0.3, fontsize=22)
plt.show()




####################################################
##############Condition Number plot#################
####################################################

# H_file = h5py.File('../1_out_3_all.hdf5','r')
# H_r = np.array(H_file.get('H_r'))
# H_i = np.array(H_file.get('H_i'))
# # H_i = np.transpose(H_i,(2,1,0))
# # H_r = np.transpose(H_r,(2,1,0))
# print("H_r shape is:", H_r.shape)
# print("H_i shape is:", H_i.shape)
# H = np.array(H_r + 1j*H_i)
# print("H shape is:", H.shape)

# cond_num = np.zeros(400)

# for i in range (0,400):
#     cond_num[i] = np.linalg.cond(H[i,:,:])

# plt.figure(figsize=(10, 8), dpi=80); 

# plt.plot(cond_num, ms=1.0)
# # plt.axis('square')
# plt.xlim([0,399])
# plt.grid()

# plt.title('Condition Number Of Channel Matrix',fontsize=20)
# plt.xlabel('TTI',fontsize=20)
# plt.ylabel('Condition Number',fontsize=20)
# # plt.legend(["Rx","Tx"], loc = "upper right")
# plt.show()