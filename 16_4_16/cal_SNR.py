from audioop import avg
import numpy as np 
import matplotlib.pyplot as plt 
import h5py

H_file = h5py.File('./6416_channel.hdf5','r')
H_r = np.array(H_file.get('H_r'))
H_i = np.array(H_file.get('H_i'))
H_i = np.transpose(H_i,(2,1,0))
H_r = np.transpose(H_r,(2,1,0))
print("H_r shape is:", H_r.shape)
print("H_i shape is:", H_i.shape)
H = np.array(H_r[0:400,:,:] + 1j*H_i[0:400,:,:])
print("H shape is:", H.shape)
# (400,16=M,16=K)
# SNR = 10* log10 (squeeze(min( abs ( H (1, : ,:) ) .^2 ,[],2))./(1e-2))

SNR = np.zeros((400,16))
for i in range (0,400):
    for j in range (0,16):
        SNR[i,j] = 10 * np.log10(1e3*np.min(np.squeeze(np.abs(H[i,:,j])**2))/(1e-2))

avg_SNR = np.zeros((16,))
avg_SNR = np.mean(SNR,axis=0)
print("avg_SNR:", avg_SNR) 

# plt.figure(figsize=(10, 8), dpi=80); 

# plt.plot(SNR[:,1],linewidth=2)
# plt.plot(SNR[:,4],linewidth=2)
# plt.plot(SNR[:,7],linewidth=2)
# plt.xlim([0,399])
# plt.grid()

# plt.title('UEs SNR at Each TTI in Mobile Scenario',fontsize=20)
# plt.xlabel('TTI',fontsize=18)
# plt.ylabel('SNR',fontsize=18)
# plt.yticks(fontsize = 18)
# plt.xticks(fontsize = 18)
# plt.legend(["UE2","UE4","UE7"], loc = 'lower right',fancybox=True, framealpha=0.3, fontsize=18)
# plt.show()