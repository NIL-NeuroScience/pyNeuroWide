"""
MAIN_Rbp4.py

Rbp4 data analysis script.

author: Bradley Rauscher (June, 2026)
"""
# %%
import numpy as np
from pyNeuroWide import IRF, colors, plot, spectra
from pyNeuroWide.process import bpf
from matplotlib import pyplot as plt

# %% load data
Rbp4_path = "/Users/bcraus/Library/Mobile Documents/com~apple~CloudDocs/Documents/BostonU/Research/DevorLab/code/datasets/Rbp4_data.npy"
data_Rbp4 = np.load(Rbp4_path, allow_pickle=True).item()

# generate subject summary
def log_summary(log):
    order = []
    mice = [i['Mouse'] for i in log]
    unique_mice = np.unique(mice)
    M = len(unique_mice)

    for idx in range(M):
        tmp = {}
        tmp['Mouse'] = unique_mice[idx]
        tmp['Runs'] = [i for i,m in enumerate(mice) if m==unique_mice[idx]]
        order.append(tmp)
    return order

# process data

def data_processing(data):
    N = len(data['log'])

    data['raw_HbT'] = data['HbT'].copy()

    for i in range(N):
        data['HbT'][i] = bpf(data['HbT'][i], [0,0.5], fs=10)
    
    return data

data_Rbp4 = data_processing(data_Rbp4)

# %% analyze data

def data_analysis(data):
    N = len(data['log'])
    order = log_summary(data['log'])
    M = len(order)
    
    SSp_mask = [False] * 12
    SSp_mask[3:5] = [True] * 2
    
    r_Ca_HbT = np.zeros((12,N))
    IRF_global_r = np.zeros((12,N))
    IRF_global = np.zeros((101,N))
    IRF_SSp_r = np.zeros((12,N))
    IRF_SSp = np.zeros((101,N))
    IRF_var_r = np.zeros((12,N))
    IRF_var = np.zeros((101,12,N))
    spc_Ca = np.zeros((4097,12,N))
    spc_HbT = np.zeros((4097,12,N))
    coh_Ca_HbT = np.zeros((4097,12,N))
    phi_Ca_HbT = np.zeros((4097,12,N))
    xc_Ca_HbT = np.zeros((101,12,N))

    subAvg = {}
    subAvg['r_Ca_HbT'] = np.zeros((12,M))
    subAvg['IRF_global_r'] = np.zeros((12,M))
    subAvg['IRF_global'] = np.zeros((101,M))
    subAvg['IRF_SSp_r'] = np.zeros((12,M))
    subAvg['IRF_SSp'] = np.zeros((101,M))
    subAvg['IRF_var_r'] = np.zeros((12,M))
    subAvg['IRF_var'] = np.zeros((101,12,M))
    subAvg['spc_Ca'] = np.zeros((4097,12,M))
    subAvg['spc_HbT'] = np.zeros((4097,12,M))
    subAvg['coh_Ca_HbT'] = np.zeros((4097,12,M))
    subAvg['phi_Ca_HbT'] = np.zeros((4097,12,M))
    subAvg['xc_Ca_HbT'] = np.zeros((101,12,M))

    for i in range(N):
        r_Ca_HbT[:,i] = IRF.corr(data['HbT'][i], data['Ca'][i])
        IRF_global_r[:,i],IRF_global[:,i],_ = IRF.IRFx1(data['HbT'][i], data['Ca'][i], [0,100], [True] * 12)
        IRF_SSp_r[:,i],IRF_SSp[:,i],_ = IRF.IRFx1(data['HbT'][i], data['Ca'][i], [0,100], SSp_mask)
        # IRF_var_r[:,i],IRF_var[:,:,i],_ = IRF.IRFx1_varWeights(data['HbT'][i], data['Ca'][i], [0,100])
        spc_Ca[:,:,i], fr = spectra.mtspectrumc(data['Ca'][i], tapers=[5,9], Fs=10, T=6000)
        spc_HbT[:,:,i], fr = spectra.mtspectrumc(data['raw_HbT'][i], tapers=[5,9], Fs=10, T=6000)
        coh_Ca_HbT[:,:,i], phi_Ca_HbT[:,:,i], fr = spectra.coherencyc(data['raw_HbT'][i], data['Ca'][i], tapers=[5,9], Fs=10, T=6000)
        xc_Ca_HbT[:,:,i] = IRF.xcorr(data['Ca'][i], data['HbT'][i], maxlag=50)

    spc_Ca = spc_Ca / spc_Ca.sum(axis=0, keepdims=True)
    spc_HbT = spc_HbT / spc_HbT.sum(axis=0, keepdims=True)
    
    for i in range(M):
        subAvg['r_Ca_HbT'][:,i] = r_Ca_HbT[:,order[i]['Runs']].mean(axis=1)
        subAvg['IRF_global_r'][:,i] = IRF_global_r[:,order[i]['Runs']].mean(axis=1)
        subAvg['IRF_global'][:,i] = IRF_global[:,order[i]['Runs']].mean(axis=1)
        subAvg['IRF_SSp_r'][:,i] = IRF_SSp_r[:,order[i]['Runs']].mean(axis=1)
        subAvg['IRF_SSp'][:,i] = IRF_SSp[:,order[i]['Runs']].mean(axis=1)
        # subAvg['IRF_var_r'][:,i] = IRF_var_r[:,order[i]['Runs']].mean(axis=1)
        # subAvg['IRF_var'][:,:,i] = IRF_var[:,:,order[i]['Runs']].mean(axis=2)
        subAvg['spc_Ca'][:,:,i] = spc_Ca[:,:,order[i]['Runs']].mean(axis=2)
        subAvg['spc_HbT'][:,:,i] = spc_HbT[:,:,order[i]['Runs']].mean(axis=2)
        subAvg['coh_Ca_HbT'][:,:,i] = coh_Ca_HbT[:,:,order[i]['Runs']].mean(axis=2)
        subAvg['phi_Ca_HbT'][:,:,i] = phi_Ca_HbT[:,:,order[i]['Runs']].mean(axis=2)
        subAvg['xc_Ca_HbT'][:,:,i] = xc_Ca_HbT[:,:,order[i]['Runs']].mean(axis=2)

    subAvg['fr'] = fr
    subAvg['M'] = M

    return subAvg

subAvg_Rbp4 = data_analysis(data_Rbp4)

# %% 
# plot signals

run_idx = 15
t = np.arange(0, 600, 0.1) / 60

plt.rcParams.update({"font.size": 12})

fig, axes = plt.subplots(5, 1, figsize=(10,6))

axes[0].plot(t, data_Rbp4['Ca'][run_idx][:,4], color=colors.r(), linewidth=1)
axes[1].plot(t, data_Rbp4['HbT'][run_idx][:,4] - 10, color=colors.b(), linewidth=1)
axes[2].plot(t, data_Rbp4['Pupil'][run_idx], color=colors.p(), linewidth=1)
axes[3].plot(t, data_Rbp4['Whisking'][run_idx], color=colors.c(), linewidth=1)
axes[4].plot(t, data_Rbp4['Acc'][run_idx], color=[0,0,0], linewidth=1)

[axes[i].set_xlim(0, 10) for i in range(5)]
[axes[i].set_yticks([]) for i in range(5)]
[axes[i].set_xticks([]) for i in range(4)]
[axes[i].spines["top"].set_visible(False) for i in range(5)]
[axes[i].spines["right"].set_visible(False) for i in range(5)]
[axes[i].spines["left"].set_visible(False) for i in range(5)]
[axes[i].spines["bottom"].set_visible(False) for i in range(4)]
[axes[i].set_ylabel(n) for i,n in enumerate(['Ca2+','HbT','Pupil','Whisking','Movement'])]
axes[4].set_xlabel("Time (min)")
plt.show()

# %% 
# plot IRF performance
fig, axes = plt.subplots(1, 4, figsize=(10,3))

plot.allenMap(axes[0], subAvg_Rbp4['r_Ca_HbT'].mean(axis=1), clim=[0,1], clabel='r', side='left', title='r')
plot.allenMap(axes[1], subAvg_Rbp4['IRF_global_r'].mean(axis=1), clim=[0,1], clabel='r', side='left', title='global')
plot.allenMap(axes[2], subAvg_Rbp4['IRF_SSp_r'].mean(axis=1), clim=[0,1], clabel='r', side='left', title='SSp')
plot.allenMap(axes[3], subAvg_Rbp4['IRF_var_r'].mean(axis=1), clim=[0,1], clabel='r', side='left', title='var')
plt.show()

# %%
# plot spectra and coherence

fig, axes = plt.subplots(1, 2, figsize=(8,3))

plot.lineError(axes[0],
               subAvg_Rbp4['fr'], 
               subAvg_Rbp4['fr'] * subAvg_Rbp4['spc_Ca'].mean(axis=(1,2)), 
               subAvg_Rbp4['fr'] * subAvg_Rbp4['spc_Ca'].mean(axis=1).std(axis=1) / np.sqrt(subAvg_Rbp4['M']))
plot.lineError(axes[0],
               subAvg_Rbp4['fr'], 
               subAvg_Rbp4['fr'] * subAvg_Rbp4['spc_HbT'].mean(axis=(1,2)), 
               subAvg_Rbp4['fr'] * subAvg_Rbp4['spc_HbT'].mean(axis=1).std(axis=1) / np.sqrt(subAvg_Rbp4['M']), 
               xscale='log', 
               ylabel='PSD * F',
               xlabel='F (Hz)',
               xlim=[0.05,5],
               color=colors.o())

plot.lineError(axes[1],
               subAvg_Rbp4['fr'], 
               subAvg_Rbp4['coh_Ca_HbT'].mean(axis=(1,2)), 
               subAvg_Rbp4['coh_Ca_HbT'].mean(axis=1).std(axis=1) / np.sqrt(subAvg_Rbp4['M']), 
               xscale='log', 
               ylabel='Coherence',
               xlabel='F (Hz)',
               xlim=[0.05,5])

[axes[i].spines["top"].set_visible(False) for i in range(2)]
[axes[i].spines["right"].set_visible(False) for i in range(2)]

plt.show()
# %%
# plot xcorr

fig, axes = plt.subplots(1, 1, figsize=(6,3))

plot.lineError(axes,
               np.arange(5,-5.1,-0.1), 
               subAvg_Rbp4['xc_Ca_HbT'].mean(axis=(1,2)), 
               subAvg_Rbp4['xc_Ca_HbT'].mean(axis=1).std(axis=1) / np.sqrt(subAvg_Rbp4['M']), 
               ylabel='r',
               xlabel='Delay (s)',
               xlim=[-5,5])

axes.spines["top"].set_visible(False)
axes.spines["right"].set_visible(False)
plt.show()