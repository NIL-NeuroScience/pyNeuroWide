"""
disinhibition.py

Disinhibition data analysis script.

author: Bradley Rauscher (September, 2026)
"""
# %%
import numpy as np
from pyNeuroWide import IRF, colors, plot, spectra
from pyNeuroWide.process import bpf
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from scipy.signal import detrend, find_peaks

# %% load data
data_path = "/Users/bcraus/Library/Mobile Documents/com~apple~CloudDocs/Documents/BostonU/Research/DevorLab/code/datasets/disinhibition.npy"
data = np.load(data_path, allow_pickle=True)
gfp = [d['gfp'] for d in data]
data_Rbp4 = data[[i for i, x in enumerate(gfp) if x == "Rbp4"]]
data_VIP = data[[i for i, x in enumerate(gfp) if x == "VIP"]]
data_Calb = data[[i for i, x in enumerate(gfp) if x == "Calb"]]

# %% define save paths
fig_1 = "/Users/bcraus/Library/Mobile Documents/com~apple~CloudDocs/Documents/BostonU/Research/DevorLab/Regulation of CBF by PNs/Figures/Figure_1/"
fig_2 = "/Users/bcraus/Library/Mobile Documents/com~apple~CloudDocs/Documents/BostonU/Research/DevorLab/Regulation of CBF by PNs/Figures/Figure_2/"
fig_3 = "/Users/bcraus/Library/Mobile Documents/com~apple~CloudDocs/Documents/BostonU/Research/DevorLab/Regulation of CBF by PNs/Figures/Figure_3/"
fig_4 = "/Users/bcraus/Library/Mobile Documents/com~apple~CloudDocs/Documents/BostonU/Research/DevorLab/Regulation of CBF by PNs/Figures/Figure_4/"
sfig_1 = "/Users/bcraus/Library/Mobile Documents/com~apple~CloudDocs/Documents/BostonU/Research/DevorLab/Regulation of CBF by PNs/Figures/Suppl_Figure_1/"

# %% data processing

def data_processing(data):
    N = len(data)

    for i in range(N):
        # detrend signals
        data[i]['parcells']['HbT'] = detrend(data[i]['parcells']['HbT'], type="linear", axis=0)
        data[i]['parcells_all']['HbT'] = detrend(data[i]['parcells_all']['HbT'], type="linear", axis=0)

        data[i]['parcells']['gfp_HD'] = detrend(data[i]['parcells']['gfp_HD'], type="linear", axis=0)
        data[i]['parcells_all']['gfp_HD'] = detrend(data[i]['parcells_all']['gfp_HD'], type="linear", axis=0)

        # low-pass filter HbT
        data[i]['parcells']['raw_HbT'] = data[i]['parcells']['HbT'].copy()
        data[i]['parcells_all']['raw_HbT'] = data[i]['parcells_all']['HbT'].copy()

        data[i]['parcells']['HbT'] = bpf(data[i]['parcells']['HbT'], [0,0.5], fs=10)
        data[i]['parcells_all']['HbT'] = bpf(data[i]['parcells_all']['HbT'], [0,0.5], fs=10)
    
    return data

data_Calb = data_processing(data_Calb)
data_VIP = data_processing(data_VIP)
data_Rbp4 = data_processing(data_Rbp4)

# %% organize data

def create_log(data):
    log = {}
    log['Mouse'] = [d['Mouse'] for d in data]
    log['Date'] = [d['Date'] for d in data]
    log['Run'] = [d['Run'] for d in data]
    log['Type'] = [d['Type'] for d in data]

    order = {}
    order['Mouse'] = np.unique(log['Mouse'])
    order['Runs'] = []
    
    M = len(order['Mouse'])

    for idx in range(M):
        order['Runs'].append([i for i,m in enumerate([d['Mouse'] for d in data]) if m==order['Mouse'][idx]])
    
    return pd.DataFrame(log), pd.DataFrame(order)

log_Rbp4, order_Rbp4 = create_log(data_Rbp4)
log_VIP, order_VIP = create_log(data_VIP)
log_Calb, order_Calb = create_log(data_Calb)

# %% run analysis

def analysis(data):
    log, order = create_log(data)

    type = [d['Type'] for d in data]
    stim_idx = [i for i, s in enumerate(type) if s == "baseline"]

    Mice = np.unique(log.Mouse[stim_idx])
    t_shift = np.arange(-50,101)

    subAvg = {}
    subAvg['M'] = len(Mice)
    subAvg['r'] = []
    subAvg['IRF'] = []
    subAvg['IRF_r'] = []
    subAvg['IRF_SSp'] = []
    subAvg['IRF_SSp_r'] = []
    subAvg['IRF_var'] = []
    subAvg['IRF_var_r'] = []
    subAvg['irf_P'] = []
    subAvg['irf_SSp_P'] = []
    subAvg['irf_var_P'] = []
    subAvg['xc'] = []
    subAvg['IRF2_r'] = []

    for mouse in Mice:
        session_idx = [i for i in range(len(log)) if log.Type[i] == "baseline" and log.Mouse[i] == mouse]

        r = []
        irf_r = []
        irf = []
        irf_SSp_r = []
        irf_SSp = []
        irf_var_r = []
        irf_var = []
        irf_P = []
        irf_SSp_P = []
        irf_var_P = []
        irf2_r = []
        xc = []

        doIRF = True
        
        W = 150
        for i in session_idx:
            if doIRF:
                tIRF_r,tIRF,_,tIRF_r_dt = IRF.IRFx1(data[i]['parcells']['HbT'], data[i]['parcells']['gfp_HD'], [0,100], [True] * 12, W=W)
                tIRF_SSp_r,tIRF_SSp,_,tIRF_SSp_r_dt = IRF.IRFx1(data[i]['parcells']['HbT'], data[i]['parcells']['gfp_HD'], [0,100], [False] * 3 + [True] * 2 + [False] * 7, W=W)
                tIRF_var_r,tIRF_var,_,tIRF_var_r_dt = IRF.IRFx1_varWeights(data[i]['parcells']['HbT'], data[i]['parcells']['gfp_HD'], [0,100], W=W)
                pupil = data[i]['Pupil']
                pupil = bpf(pupil,fr=[0,0.1],fs=10)
                pupil = np.concatenate((np.diff(pupil),[0]))
                # X = np.stack([data[i]['parcells']['gfp_HD'], np.tile(pupil[:,None],(1,12))],axis=2)
                # tirf2_r,_,_,_ = IRF.alphaXDeconvolve(data[i]['parcells']['HbT'], X, [0,100], W=W)
            elif i == session_idx[0]:
                tIRF_r,tIRF,_,tIRF_r_dt = IRF.IRFx1(data[i]['parcells']['HbT'], data[i]['parcells']['gfp_HD'], [0,100], [True] * 12, W=W)
                tIRF_SSp_r,tIRF_SSp,_,tIRF_SSp_r_dt = IRF.IRFx1(data[i]['parcells']['HbT'], data[i]['parcells']['gfp_HD'], [0,100], [False] * 3 + [True] * 2 + [False] * 7, W=W)
                tIRF_var_r,tIRF_var,_,tIRF_var_r_dt = IRF.IRFx1_varWeights(data[i]['parcells']['HbT'], data[i]['parcells']['gfp_HD'], [0,100], W=W)
            
            irf_r.append(tIRF_r)
            irf.append(tIRF)
            irf_SSp_r.append(tIRF_SSp_r)
            irf_SSp.append(tIRF_SSp)
            irf_var_r.append(tIRF_var_r)
            irf_var.append(tIRF_var)
            # irf2_r.append(tirf2_r)

            irf_P.append(IRF.corr(tIRF_r_dt,data[i]['Pupil'][int(W/2-1):-int(W/2),None]))
            irf_SSp_P.append(IRF.corr(tIRF_SSp_r_dt,data[i]['Pupil'][int(W/2-1):-int(W/2),None]))
            irf_var_P.append(IRF.corr(tIRF_var_r_dt,data[i]['Pupil'][int(W/2-1):-int(W/2),None]))
            r.append(IRF.corr(data[i]['parcells']['HbT'][7:], data[i]['parcells']['gfp_HD'][:-7], dim=0))

            xc.append(IRF.xcorr(data[i]['parcells']['HbT'], data[i]['parcells']['gfp_HD'], 100))

        subAvg['IRF'].append(np.stack(irf, axis=1).mean(axis=1))
        subAvg['IRF_r'].append(np.stack(irf_r, axis=1).mean(axis=1))
        # subAvg['IRF2_r'].append(np.stack(irf2_r, axis=1).mean(axis=1))
        subAvg['IRF_SSp'].append(np.stack(irf_SSp, axis=1).mean(axis=1))
        subAvg['IRF_SSp_r'].append(np.stack(irf_SSp_r, axis=1).mean(axis=1))
        subAvg['IRF_var'].append(np.stack(irf_var, axis=2).mean(axis=2))
        subAvg['IRF_var_r'].append(np.stack(irf_var_r, axis=1).mean(axis=1))
        subAvg['r'].append(np.stack(r, axis=1).mean(axis=1))
        subAvg['irf_P'].append(np.stack(irf_P, axis=1).mean(axis=1))
        subAvg['irf_SSp_P'].append(np.stack(irf_SSp_P, axis=1).mean(axis=1))
        subAvg['irf_var_P'].append(np.stack(irf_var_P, axis=1).mean(axis=1))
        subAvg['xc'].append(np.stack(xc, axis=2).mean(axis=2))

    return subAvg

subAvg_Calb = analysis(data_Calb)
subAvg_VIP = analysis(data_VIP)
subAvg_Rbp4 = analysis(data_Rbp4)

# %% stim analysis
def stim_analysis(data):
    log, order = create_log(data)

    type = [d['Type'] for d in data]
    stim_idx = [i for i, s in enumerate(type) if s == "visual stim"]

    Mice = np.unique(log.Mouse[stim_idx])
    t_shift = np.arange(-50,151)

    subAvg = {}
    subAvg['M'] = len(Mice)
    subAvg['gfp_HD'] = []
    subAvg['HbT'] = []
    subAvg['IRF'] = []
    subAvg['IRF_r'] = []
    subAvg['gfp_HD_amp'] = []
    subAvg['HbT_amp'] = []

    for mouse in Mice:
        session_idx = [i for i in range(len(log)) if log.Type[i] == "visual stim" and log.Mouse[i] == mouse]
        trials_gfp_HD = []
        trials_HbT = []
        trials_IRF = []

        for i in session_idx:
            for t in data[i]['stim']['t_idx']:
                trials_gfp_HD.append(data[i]['parcells_all']['gfp_HD'][t+t_shift,:])
                trials_HbT.append(data[i]['parcells_all']['HbT'][t+t_shift,:])

        N = len(trials_HbT)
        r,irf,_,_ = IRF.IRFx1(np.stack(trials_HbT,axis=2).mean(axis=2), np.stack(trials_gfp_HD,axis=2).mean(axis=2), [0,100], [False] * 11 + [True] + [False] * 12)

        subAvg['gfp_HD'].append(np.stack(trials_gfp_HD,axis=2).mean(axis=2))
        subAvg['HbT'].append(np.stack(trials_HbT,axis=2).mean(axis=2))
        subAvg['IRF'].append(irf)
        subAvg['IRF_r'].append(r)
        subAvg['gfp_HD_amp'].append(np.stack(trials_gfp_HD,axis=2)[50:90,11,:].mean(axis=0) - np.stack(trials_gfp_HD,axis=2)[0:50,11,:].mean(axis=0))
        subAvg['HbT_amp'].append(np.stack(trials_HbT,axis=2)[50:90,11,:].mean(axis=0) - np.stack(trials_HbT,axis=2)[0:50,11,:].mean(axis=0))

    return subAvg

stimAvg_Calb = stim_analysis(data_Calb)
stimAvg_VIP = stim_analysis(data_VIP)
stimAvg_Rbp4 = stim_analysis(data_Rbp4)

# %% plot parameters

plt.rcParams.update({"font.size": 8})
plt.rcParams['ytick.major.size'] = 2.835 / 2
plt.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['svg.fonttype'] = "none"
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 8

# %% plot figure 1

session = 7

T = data_Rbp4[session]['parcells']['gfp_HD'].shape[0]

fig = plt.figure(figsize=(8,8))

main = fig.add_gridspec(2, 2)

signals = main[0,1].subgridspec(5,1)
maps = main[1,0].subgridspec(3,2)
var_maps = main[1,1].subgridspec(3,2)

# plot IRFs and performance maps
irfs = []
r_maps = []
for i in range(3):
    irfs.append(fig.add_subplot(maps[i,0]))
    r_maps.append(fig.add_subplot(maps[i,1]))
    if i < 2:
        plot.remove_axes(irfs[i], "RBT")
    else:
        plot.remove_axes(irfs[i], "RT")
    irfs[i].set_xlim(0,5)
    irfs[i].set_ylabel('a.u.')

t_irfs = np.arange(0,10.1,0.1)
plot.lineError(irfs[0],t_irfs,np.stack(subAvg_Rbp4['IRF'],axis=1).mean(axis=1), np.stack(subAvg_Rbp4['IRF'],axis=1).std(axis=1) / np.sqrt(subAvg_Rbp4['M']),color=colors.mb(),title="global IRF",ylabel="a.u.")
plot.lineError(irfs[1],t_irfs,np.stack(subAvg_Rbp4['IRF_SSp'],axis=1).mean(axis=1), np.stack(subAvg_Rbp4['IRF_SSp'],axis=1).std(axis=1) / np.sqrt(subAvg_Rbp4['M']),color=colors.mb(),title="SSp IRF",ylabel="a.u.")
plot.lineError(irfs[2],t_irfs,np.stack(subAvg_Rbp4['IRF_var'],axis=2).mean(axis=2)[:,4], np.stack(subAvg_Rbp4['IRF_var'],axis=2).std(axis=2)[:,4] / np.sqrt(subAvg_Rbp4['M']),color=colors.mb())
plot.lineError(irfs[2],t_irfs,np.stack(subAvg_Rbp4['IRF_var'],axis=2).mean(axis=2)[:,1], np.stack(subAvg_Rbp4['IRF_var'],axis=2).std(axis=2)[:,1] / np.sqrt(subAvg_Rbp4['M']),color=colors.o(),title="variant IRF",ylabel="a.u.")
plot.allenMap(r_maps[0],np.stack(subAvg_Rbp4['IRF_r'],axis=1).mean(axis=1),clim=[0,1],side="right",clabel="r",outline=[0,1,2,3,4,5,6,7,8,9,10,11],linewidth=0.5)
plot.allenMap(r_maps[1],np.stack(subAvg_Rbp4['IRF_SSp_r'],axis=1).mean(axis=1),clim=[0,1],side="right",clabel="r",outline=[3,4],linewidth=0.5)
plot.allenMap(r_maps[2],np.stack(subAvg_Rbp4['IRF_var_r'],axis=1).mean(axis=1),clim=[0,1],side="right",clabel="r",outline=[1,3],linewidth=0.5,color=[colors.mb(),colors.o()])
irfs[2].set_xlabel('Delay (s)')

plot.remove_axes(irfs[0], "RBT")
plot.remove_axes(irfs[1], "RBT")

# plot signals
sigs = []
for i in range(5):
    sigs.append(fig.add_subplot(signals[i]))
    if i < 4:
        plot.remove_axes(sigs[i], "LRBT")
    else:
        plot.remove_axes(sigs[i], "LRT")
    sigs[i].set_xlim(0,10)

t = np.arange(0,T / 10,0.1) / 60 - 10
plot_colors = [colors.g(), colors.b(), colors.p(), colors.c(), [0,0,0]]

sigs[0].plot(t, data_Rbp4[session]['parcells']['gfp_HD'][:,4], color=plot_colors[0], linewidth=0.5)
sigs[0].plot([10,10],[0,10],color=plot_colors[0],linewidth=4)
sigs[1].plot(t, data_Rbp4[session]['parcells']['HbT'][:,4], color=plot_colors[1], linewidth=0.5)
sigs[1].plot([10,10],[0,2.5],color=plot_colors[1],linewidth=4)
sigs[2].plot(t, data_Rbp4[session]['Pupil'], color=plot_colors[2], linewidth=0.5)
sigs[2].plot([10,10],[0.3,0.5],color=plot_colors[2],linewidth=4)
sigs[3].plot(t, data_Rbp4[session]['Whisking'], color=plot_colors[3], linewidth=0.5)
sigs[4].plot(t, data_Rbp4[session]['Accelerometer'], color=plot_colors[4], linewidth=0.5)
sigs[4].set_xlabel('Time (min)')

for i, l in enumerate(["Ca2+","HbT","Pupil","Whisking","Movement"]):
    sigs[i].set_ylabel(l, color=plot_colors[i])

# var IRF maps
var_maps_ax = []
for i in range(3):
    var_maps_ax.append(fig.add_subplot(var_maps[i,0]))
    var_maps_ax.append(fig.add_subplot(var_maps[i,1]))

pos_amps = np.stack(subAvg_Rbp4['IRF_var'],axis=2).mean(axis=2).max(axis=0)
neg_amps = np.stack(subAvg_Rbp4['IRF_var'],axis=2).mean(axis=2).min(axis=0)

plot.allenMap(var_maps_ax[0],pos_amps,side="right",clim=[-pos_amps.max(),pos_amps.max()],clabel="A",cmap="bwr")
plot.allenMap(var_maps_ax[1],neg_amps,side="right",clim=[neg_amps.min(),-neg_amps.min()],clabel="B",cmap="bwr")
t_xc = np.arange(-10,10.1,0.1)
plot.lineError(var_maps_ax[2],t_xc,np.stack(subAvg_Rbp4['xc'],axis=2).mean(axis=2)[:,4], np.stack(subAvg_Rbp4['xc'],axis=2).std(axis=2)[:,5] / np.sqrt(len(subAvg_Rbp4)),color=colors.mb(),ylabel="r")
plot.lineError(var_maps_ax[2],t_xc,np.stack(subAvg_Rbp4['xc'],axis=2).mean(axis=2)[:,1], np.stack(subAvg_Rbp4['xc'],axis=2).std(axis=2)[:,2] / np.sqrt(len(subAvg_Rbp4)),color=colors.o(),ylabel="r",xlim=[-5,5],xlabel="Lag (s)")
plot.allenMap(var_maps_ax[3],t_xc[np.argmax(np.stack(subAvg_Rbp4['xc'],axis=2).mean(axis=2),axis=0)],side="right",clim=[-2,2],clabel="Lag (s)",cmap="bwr")
print(t_xc[np.argmax(np.stack(subAvg_Rbp4['xc'],axis=2).mean(axis=2),axis=0)].mean())

plot.allenMap(var_maps_ax[4],np.stack(subAvg_Rbp4['irf_SSp_P'],axis=1).mean(axis=1),clim=[-0.5,0.5],side="right",clabel="r",cmap="bwr")
perf_vals = [np.stack(subAvg_Rbp4['IRF_r'],axis=1).mean(axis=0),
             np.stack(subAvg_Rbp4['IRF_var_r'],axis=1).mean(axis=0),
             np.stack(subAvg_Calb['IRF_r'],axis=1).mean(axis=0),
             np.stack(subAvg_Calb['IRF_var_r'],axis=1).mean(axis=0)]
plot.bar(var_maps_ax[5], perf_vals, title="IRF performance", alpha=0.5, ylabel="r")
plot.remove_axes(var_maps_ax[2], "RT")

plot.prepare(fig)
fig.savefig(fig_1 + "fig.svg", format="svg", bbox_inches="tight", dpi=600)
plt.show()

# %% plot figure 2

sessions = [7,0]

T = data_Calb[sessions[0]]['parcells']['gfp_HD'].shape[0]

fig = plt.figure(figsize=(8,8))

main = fig.add_gridspec(2, 2)

signals = main[0,0].subgridspec(4,1)

t = np.arange(0,T/10,0.1) / 60
y_scale = [10,20]

sigs = []
for i, data in enumerate([data_Calb, data_VIP]):
    sigs.append(fig.add_subplot(signals[i*2]))
    sigs.append(fig.add_subplot(signals[i*2+1]))

    sigs[i*2].plot(t,data[sessions[i]]['parcells']['gfp_HD'][:,4], color=colors.g(), linewidth=0.5)
    sigs[i*2+1].plot(t,data[sessions[i]]['parcells']['HbT'][:,4], color=colors.b(), linewidth=0.5)

    sigs[i*2].plot([10,10],[0,y_scale[i]],linewidth=4,color=colors.g())
    sigs[i*2+1].plot([10,10],[0,5],color=colors.b(),linewidth=4)

    plot.remove_axes(sigs[i*2], "LRBT")
    plot.remove_axes(sigs[i*2+1], "LRT")

    sigs[i*2].set_xlim(0,10)
    sigs[i*2+1].set_ylim(-5,5)
    sigs[i*2+1].set_xlim(0,10)
    sigs[i*2+1].set_xlabel("Time (min)")

Calb = main[0,1].subgridspec(3,2)
VIP = main[1,0].subgridspec(3,2)
axes = [Calb, VIP]
titles = ["Calb","VIP"]
for a, subAvg in enumerate([subAvg_Calb, subAvg_VIP]):
    irfs = []
    r_maps = []
    for i in range(3):
        irfs.append(fig.add_subplot(axes[a][i,0]))
        r_maps.append(fig.add_subplot(axes[a][i,1]))

    t_irfs = np.arange(0,10.1,0.1)
    plot.lineError(irfs[0], t_irfs, np.stack(subAvg['IRF'],axis=1).mean(axis=1), np.stack(subAvg['IRF'],axis=1).std(axis=1) / np.sqrt(subAvg['M']),
                   color=colors.mb(),
                   title="global IRF",
                   xlim=[0,5],
                   ylabel="a.u.")
    plot.lineError(irfs[1], t_irfs, np.stack(subAvg['IRF_SSp'],axis=1).mean(axis=1), np.stack(subAvg['IRF_SSp'],axis=1).std(axis=1) / np.sqrt(subAvg['M']),
                   color=colors.mb(),
                   title="SSp IRF",
                   xlim=[0,5],
                   ylabel="a.u.")
    plot.lineError(irfs[2], t_irfs, np.stack(subAvg['IRF_var'],axis=2).mean(axis=2)[:,4], np.stack(subAvg['IRF_var'],axis=2).std(axis=2)[:,4] / np.sqrt(subAvg['M']),
                   color=colors.mb())
    plot.lineError(irfs[2], t_irfs, np.stack(subAvg['IRF_var'],axis=2).mean(axis=2)[:,1], np.stack(subAvg['IRF_var'],axis=2).std(axis=2)[:,1] / np.sqrt(subAvg['M']),
                   color=colors.o(),
                   title="SSp var",
                   xlim=[0,5],
                   xlabel="Delay (s)",
                   ylabel="a.u.")
    plot.allenMap(r_maps[0], np.stack(subAvg['IRF_r'],axis=1).mean(axis=1),
                  clim=[0,1],
                  side="right",
                  clabel="r",
                  outline=[0,1,2,3,4,5,6,7,8,9,10,11],
                  linewidth=0.5)
    plot.allenMap(r_maps[1], np.stack(subAvg['IRF_SSp_r'],axis=1).mean(axis=1),
                  clim=[0,1],
                  side="right",
                  clabel="r",
                  outline=[3,4],
                  linewidth=0.5)
    plot.allenMap(r_maps[2], np.stack(subAvg['IRF_var_r'],axis=1).mean(axis=1),
                  clim=[0,1],
                  side="right",
                  clabel="r",
                  outline=[1,3],
                  linewidth=0.5,
                  color=[colors.mb(),colors.o()])
    r_maps[0].set_title(titles[a])
    plot.remove_axes(irfs[0], "RBT")
    plot.remove_axes(irfs[1], "RBT")
    plot.remove_axes(irfs[2], "RT")

# BOTTOM RIGHT PANEL
XC = main[1,1].subgridspec(3,3)

# plot cross-correlation
xc = [fig.add_subplot(XC[0,i]) for i in range(3)]
t_xc = np.arange(-10,10.1,0.1)
plot.lineError(xc[0],t_xc,np.stack(subAvg_Calb['xc'],axis=2).mean(axis=2)[:,4], np.stack(subAvg_Calb['xc'],axis=2).std(axis=2)[:,5] / np.sqrt(subAvg_Calb['M']),color=colors.mb(),ylabel="r")
plot.lineError(xc[0],t_xc,np.stack(subAvg_VIP['xc'],axis=2).mean(axis=2)[:,4], np.stack(subAvg_VIP['xc'],axis=2).std(axis=2)[:,5] / np.sqrt(subAvg_VIP['M']),color=colors.o(),ylabel="r",xlim=[-5,5],xlabel="Lag (s)")
plot.lineError(xc[1],t_xc,np.stack(subAvg_Calb['xc'],axis=2).mean(axis=2)[:,1], np.stack(subAvg_Calb['xc'],axis=2).std(axis=2)[:,2] / np.sqrt(subAvg_Calb['M']),color=colors.mb(),ylabel="r")
plot.lineError(xc[1],t_xc,np.stack(subAvg_VIP['xc'],axis=2).mean(axis=2)[:,1], np.stack(subAvg_VIP['xc'],axis=2).std(axis=2)[:,2] / np.sqrt(subAvg_VIP['M']),color=colors.o(),ylabel="r",xlim=[-5,5],xlabel="Lag (s)")
[plot.remove_axes(xc[i],"RT") for i in range(2)]
xc[0].set_title("SSp-ll")
xc[1].set_title("MOs")

reg = 2
perf_vals = [np.stack(subAvg_Rbp4['IRF_r'],axis=1).mean(axis=0),
             np.stack(subAvg_Calb['IRF_r'],axis=1).mean(axis=0),
             np.stack(subAvg_VIP['IRF_r'],axis=1).mean(axis=0)]
plot.bar(xc[2], perf_vals, title="IRF performance", labels=["Rbp4","Calb","VIP"], alpha=0.5, ylabel="r", color=[colors.g(), colors.o(), colors.p()])
plot.remove_axes(xc[2], "RT")

beh_r = [fig.add_subplot(XC[i+1,0]) for i in range(2)]
t_xc = np.arange(-10,10.1,0.1)
plot.allenMap(beh_r[0],t_xc[np.argmax(np.stack(subAvg_Calb['xc'],axis=2).mean(axis=2),axis=0)],side="right",clim=[-2,2],clabel="Lag (s)",cmap="bwr")
plot.allenMap(beh_r[1],t_xc[np.argmax(np.stack(subAvg_VIP['xc'],axis=2).mean(axis=2),axis=0)],side="right",clim=[-2,2],clabel="Lag (s)",cmap="bwr")
print(t_xc[np.argmax(np.stack(subAvg_Calb['xc'],axis=2).mean(axis=2),axis=0)].mean())
print(t_xc[np.argmax(np.stack(subAvg_VIP['xc'],axis=2).mean(axis=2),axis=0)].mean())

# plot.allenMap(beh_r[0],np.stack(subAvg_Calb['irf_SSp_P'],axis=1).mean(axis=1),clim=[-0.5,0.5],side="right",clabel="r",cmap="bwr",title="Calb")
# plot.allenMap(beh_r[1],np.stack(subAvg_VIP['irf_SSp_P'],axis=1).mean(axis=1),clim=[-0.5,0.5],side="right",clabel="r",cmap="bwr",title="VIP")

var_Calb = [fig.add_subplot(XC[1,i+1]) for i in range(2)]
pos_amps = np.stack(subAvg_Calb['IRF_var'],axis=2).mean(axis=2).max(axis=0)
neg_amps = np.stack(subAvg_Calb['IRF_var'],axis=2).mean(axis=2).min(axis=0)
plot.allenMap(var_Calb[0],pos_amps,side="right",clim=[-pos_amps.max(),pos_amps.max()],clabel="A",cmap="bwr")
plot.allenMap(var_Calb[1],neg_amps,side="right",clim=[neg_amps.min(),-neg_amps.min()],clabel="B",cmap="bwr")

var_VIP = [fig.add_subplot(XC[2,i+1]) for i in range(2)]
pos_amps = np.stack(subAvg_VIP['IRF_var'],axis=2).mean(axis=2).max(axis=0)
neg_amps = np.stack(subAvg_VIP['IRF_var'],axis=2).mean(axis=2).min(axis=0)
plot.allenMap(var_VIP[0],pos_amps,side="right",clim=[-pos_amps.max(),pos_amps.max()],clabel="A",cmap="bwr")
plot.allenMap(var_VIP[1],neg_amps,side="right",clim=[neg_amps.min(),-neg_amps.min()],clabel="B",cmap="bwr")

plot.prepare(fig)
fig.savefig(fig_2 + "fig.svg", format="svg", bbox_inches="tight", dpi=600)
plt.show()

# %% plot figure 3

def find_duration(data, p):
    data = data - data[0:50,:].mean(axis=0)
    data = data[50::,:]
    th = data.max(axis=0) * p

    bin = np.diff(np.array(data > th,dtype=int), axis=0)
    N = data.shape[1]

    durations = []
    for i in range(N):
        up = np.where(bin[:,i] == 1)[0]
        down = np.where(bin[:,i] == -1)[0]

        if down.size == 0:
            down = data.shape[0]

        durations.append((down[0] - up[0]) / 10)

    return durations

t = np.arange(-5,15.1,0.1)

fig, axes = plt.subplots(4, 4, figsize=(8,6))
titles = ["Rbp4","Calb","VIP"]
durations = []
spatial_r = []

for i, subAvg in enumerate([stimAvg_Rbp4, stimAvg_Calb, stimAvg_VIP]):
    HbT_scale = 5
    plot.lineError(axes[i,2],t,np.stack(subAvg['gfp_HD'],axis=2).mean(axis=2)[:,11] - np.stack(subAvg['gfp_HD'],axis=2).mean(axis=2)[0:50,:].mean(axis=0)[11],np.stack(subAvg['gfp_HD'],axis=2).std(axis=2)[:,11] / np.sqrt(subAvg['M']), color=colors.g())
    plot.lineError(axes[i,2],t,HbT_scale*np.stack(subAvg['HbT'],axis=2).mean(axis=2)[:,11] - np.stack(subAvg['HbT'],axis=2).mean(axis=2)[0:50,:].mean(axis=0)[11],HbT_scale*np.stack(subAvg['HbT'],axis=2).std(axis=2)[:,11] / np.sqrt(subAvg['M']), color=colors.b())
    axes[i,2].plot([t[0],t[-1]],[0,0],ls=":",color="black")
    axes[i,2].set_ylim(-2,12)

    if i == 0:
        t1 = "Ca2+"
        t2 = "HbT"
    else:
        t1 = None
        t2 = None
    gfp_img = np.stack(subAvg['gfp_HD'],axis=2).mean(axis=2)[50:90,:].max(axis=0) - np.stack(subAvg['gfp_HD'],axis=2).mean(axis=2)[0:50,:].mean(axis=0)
    HbT_img = np.stack(subAvg['HbT'],axis=2).mean(axis=2)[50:90,:].max(axis=0) - np.stack(subAvg['HbT'],axis=2).mean(axis=2)[0:50,:].mean(axis=0)
    plot.allenMap(axes[i,0],gfp_img,clim=[0,gfp_img.max()],title=t1,clabel="∆F/F")
    plot.allenMap(axes[i,1],HbT_img,clim=[0,1],title=t2,clabel="µM")
    axes[i,2].axvspan(0,1.667,alpha=0.1, color=[0,0,0], linewidth=0)
    axes[i,2].set_xlim(-5, 15)
    axes[i,2].set_xticks([])
    axes[i,0].set_ylabel(titles[i])
    plot.remove_axes(axes[i,2], "LRBT")

    durations.append(find_duration(np.stack(subAvg['gfp_HD'],axis=2)[:,11], 0.1))
    gfp_map_ind = np.stack(subAvg['gfp_HD'],axis=2)[50:90,:,:].max(axis=0) - np.stack(subAvg['gfp_HD'],axis=2)[0:50,:,:].mean(axis=0)
    HbT_map_ind = np.stack(subAvg['HbT'],axis=2)[50:90,:,:].max(axis=0) - np.stack(subAvg['HbT'],axis=2)[0:50,:,:].mean(axis=0)
    spatial_r.append(IRF.corr(gfp_map_ind,HbT_map_ind,dim=0))

    stats = plot.scatter(axes[3,i],subAvg['gfp_HD_amp'],subAvg['HbT_amp'],
        alpha=0.3,
        xlabel="Ca2+ (% ∆F/F)",
        ylabel="HbT (µM)",
        cmp=mpl.colormaps["inferno"](np.linspace(0,0.8,3)),
        size=5)
    axes[3,i].set_title(f"r = {stats[2].mean():.2}")

axes[2,2].plot([5,9],[4,4],color="black")
axes[2,2].plot([5,5],[4,9],color="black")
axes[0,2].legend(handles=[l for l in axes[0,2].lines][0:2],labels=["Ca2+","HbT"],frameon=False)

plot.bar(axes[0,3],durations,
         title="Duration",
         color=[colors.g(),colors.o(),colors.p()],
         linewidth=1,
         labels=["1","2","3"],
         ylabel="s",
         width=0.6,
         alpha=0.5,
         size=25)

plot.bar(axes[1,3],spatial_r,
         title="Spatial Correlation",
         color=[colors.g(),colors.o(),colors.p()],
         linewidth=1,
         labels=["1","2","3"],
         ylabel="r",
         width=0.6,
         alpha=0.5,
         size=25)

Rbp4_img = np.stack(stimAvg_Rbp4['gfp_HD'],axis=2).mean(axis=2)[50:90,:].max(axis=0) - np.stack(stimAvg_Rbp4['gfp_HD'],axis=2).mean(axis=2)[0:50,:].mean(axis=0)
Calb_img = np.stack(stimAvg_Calb['gfp_HD'],axis=2).mean(axis=2)[50:90,:].max(axis=0) - np.stack(stimAvg_Calb['gfp_HD'],axis=2).mean(axis=2)[0:50,:].mean(axis=0)
VIP_img = np.stack(stimAvg_VIP['gfp_HD'],axis=2).mean(axis=2)[50:90,:].max(axis=0) - np.stack(stimAvg_VIP['gfp_HD'],axis=2).mean(axis=2)[0:50,:].mean(axis=0)
spatial_r_Ca = [[IRF.corr(Rbp4_img,Calb_img,dim=0)],
                [IRF.corr(Rbp4_img,VIP_img,dim=0)],
                [IRF.corr(VIP_img,Calb_img,dim=0)]]

plot.bar(axes[2,3],spatial_r_Ca,
         title="Spatial Correlation",
         color="black",
         linewidth=1,
         labels=["1","2","3"],
         ylabel="r",
         width=0.6,
         ylim=[0,1],
         alpha=0.5,
         size=25)

plot.remove_axes(axes[3,3],"LRTB")

# plot.prepare(fig)
fig.savefig(fig_3 + "fig.svg", format="svg", bbox_inches="tight", dpi=600)
plt.show()

# %% plot figure 4

def FC_analysis(data):
    log, order = create_log(data)

    type = [d['Type'] for d in data]
    Mice = np.unique(log.Mouse)

    w = 300
    
    subAvg = {}
    subAvg['M'] = len(Mice)

    subAvg['FC_gfp'] = []
    subAvg['FC_HbT'] = []
    subAvg['FC_HbT_high'] = []
    subAvg['FC_HbT_low'] = []
    subAvg['FC_gfp_high'] = []
    subAvg['FC_gfp_low'] = []
    subAvg['r'] = []
    
    for mouse in Mice:
        session_idx = [i for i in range(len(log)) if log.Type[i] == "baseline" and log.Mouse[i] == mouse]

        FC_gfp = []
        FC_HbT = []
        FC_HbT_high = []
        FC_HbT_low = []
        FC_gfp_high = []
        FC_gfp_low = []
        r = []

        for i in session_idx:
            FC_gfp.append(np.corrcoef(data[i]['parcells']['gfp_HD'],rowvar=False))
            FC_HbT.append(np.corrcoef(data[i]['parcells']['HbT'],rowvar=False))
            FC_gfp_dt = IRF.FC_dt(data[i]['parcells']['gfp_HD'],w)[0]
            FC_HbT_dt = IRF.FC_dt(data[i]['parcells']['HbT'],w)[0]
            r.append(IRF.corr(FC_gfp_dt,FC_HbT_dt,dim=0))

            pupil = data[i]['Pupil'][w//2-1:-w//2]
            pupil_bounds = np.percentile(pupil,(30,70))
            idx_high = pupil > pupil_bounds[1]
            idx_low = pupil < pupil_bounds[0]
            FC_gfp_low.append(FC_gfp_dt[idx_low].mean(axis=0))
            FC_gfp_high.append(FC_gfp_dt[idx_high].mean(axis=0))
            FC_HbT_low.append(FC_HbT_dt[idx_low].mean(axis=0))
            FC_HbT_high.append(FC_HbT_dt[idx_high].mean(axis=0))

        subAvg['FC_gfp'].append(np.stack(FC_gfp, axis=2).mean(axis=2))
        subAvg['FC_HbT'].append(np.stack(FC_HbT, axis=2).mean(axis=2))
        subAvg['FC_gfp_low'].append(np.stack(FC_gfp_low, axis=2).mean(axis=2))
        subAvg['FC_gfp_high'].append(np.stack(FC_gfp_high, axis=2).mean(axis=2))
        subAvg['FC_HbT_low'].append(np.stack(FC_HbT_low, axis=2).mean(axis=2))
        subAvg['FC_HbT_high'].append(np.stack(FC_HbT_high, axis=2).mean(axis=2))
        subAvg['r'].append(np.stack(r, axis=2).mean(axis=2))
        
    return subAvg

FC_Calb = FC_analysis(data_Calb)
FC_VIP = FC_analysis(data_VIP)
FC_Rbp4 = FC_analysis(data_Rbp4)

# PLOT

regions = ["MOp","MOs","SSp-bfd","SSp-tr","SSp-ll","SSp-ul","SSp-un","VISpm","VISrl","VISam","VISa","VISp"]
titles = ["Rbp4","Calb","VIP"]
fig, ax = plt.subplots(5,3,figsize=(8,14))

for i, subAvg in enumerate([FC_Rbp4, FC_Calb, FC_VIP]):
    if i == 0:
        ylabel = regions
    else:
        ylabel = None

    plot.FCmap(ax[0,i], np.stack(subAvg['FC_gfp'],axis=2).mean(axis=2),clim=[0,1],clabel="r",ylabel=ylabel,title=titles[i])
    plot.FCmap(ax[1,i], np.stack(subAvg['FC_HbT'],axis=2).mean(axis=2),clim=[0,1],clabel="r",ylabel=ylabel,title="HbT")

    plot.FCmap(ax[2,i], np.stack(subAvg['r'],axis=2).mean(axis=2),clim=[-1,1],clabel="r",ylabel=ylabel,title='HbT vs. Ca',cmap="bwr")
    plot.FCmap(ax[3,i], np.stack(subAvg['FC_gfp_high'],axis=2).mean(axis=2)-np.stack(subAvg['FC_gfp_low'],axis=2).mean(axis=2),clim=[-0.25,0.25],clabel="∆r",ylabel=ylabel,title='Ca (high-low)',cmap="bwr")
    plot.FCmap(ax[4,i], np.stack(subAvg['FC_HbT_high'],axis=2).mean(axis=2)-np.stack(subAvg['FC_HbT_low'],axis=2).mean(axis=2),clim=[-0.1,0.1],clabel="∆r",ylabel=ylabel,title='HbT (high-low)',cmap="bwr")

# %% plot figure 5
def peak_analysis(data):
    log, order = create_log(data)

    Mice = np.unique(log.Mouse)

    subAvg = {}
    subAvg['M'] = len(Mice)

    subAvg['gfp_HD'] = []
    subAvg['HbT'] = []
    subAvg['Pupil'] = []

    subAvg['prom'] = []
    subAvg['base'] = []

    subAvg['h_gfp_HD'] = []
    subAvg['h_HbT'] = []

    w = 100
    t_shift = np.arange(-w,w+1)
    
    for mouse in Mice:
        session_idx = [i for i in range(len(log)) if log.Type[i] == "baseline" and log.Mouse[i] == mouse]

        gfp_HD = []
        HbT = []
        Pupil = []
        prom = []
        h_gfp_HD = []
        h_HbT = []

        for i in session_idx:
            region = 4
            signal = data[i]['parcells']['HbT'][:,region]
            # signal = data[i]['parcells']['HbT'].mean(axis=1)
            signal = bpf(signal, fr=[0,0.1], fs=10)
            
            bounds = np.percentile(signal,(1,99))
            peaks, prop = find_peaks(
                signal,
                prominence=(bounds[1]-bounds[0]) / 10,
                distance=1
            )

            # trim peaks
            N = data[i]['parcells']['gfp_HD'].shape[0]

            prom.append(np.array([x for i,x in enumerate(prop['prominences']) if peaks[i] >= w and peaks[i] < N-w]))
            peaks = [x for x in peaks if x >= w and x < N-w]

            h_gfp_HD.append(data[i]['parcells']['gfp_HD'][peaks,region])
            h_HbT.append(data[i]['parcells']['HbT'][peaks,region])

            for t in peaks:
                gfp_HD.append(data[i]['parcells_all']['gfp_HD'][t+t_shift,:])
                HbT.append(data[i]['parcells_all']['HbT'][t+t_shift,:])
                Pupil.append(data[i]['Pupil'][t+t_shift])
        
        subAvg['gfp_HD'].append(np.stack(gfp_HD,axis=2))
        subAvg['HbT'].append(np.stack(HbT,axis=2))
        subAvg['Pupil'].append(np.stack(Pupil,axis=1))
        subAvg['prom'].append(np.concat(prom))
        subAvg['h_gfp_HD'].append(np.concat(h_gfp_HD))
        subAvg['h_HbT'].append(np.concat(h_HbT))
        subAvg['base'].append(np.concat(h_gfp_HD)-np.concat(prom))
        
    return subAvg

peaks_Calb = peak_analysis(data_Calb)
peaks_VIP = peak_analysis(data_VIP)
peaks_Rbp4 = peak_analysis(data_Rbp4)

def onsetDelay(data, baseline=30, bounds=(0.2,0.8)):
    data = data - data[0:baseline].mean(axis=0)
    data = data / data.max()
    N = np.argmax(data)
    data = data[N::-1]

    lag = []
    for i in range(2):
        idx = np.flatnonzero(data < bounds[i])[0]
        idx = idx - 1 + (data[idx-1] - bounds[i]) / (data[idx-1] - data[idx])
        lag.append(idx)

    lag = [N - lag[i] for i in range(2)]

    m = (bounds[1] - bounds[0]) / (lag[1] - lag[0])
    b = bounds[0] - lag[0] * m

    lag = -b / m

    return lag

# PLOTTING
n_bins = 3
cmp = mpl.colormaps['viridis'](np.linspace(0,1,n_bins+1))

region = 4

fig, ax = plt.subplots(4,4,figsize=(8,6))

t = np.arange(-10,10.1,0.1)
xlim = 10

t_xc = np.arange(-10,10.1,0.1)
spatial_r = []
delays = []
types = ["Rbp4","Calb","VIP"]

for i, subAvg in enumerate([peaks_Rbp4, peaks_Calb, peaks_VIP]):
    plot.lineError(ax[i,0],t,np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['gfp_HD']],axis=2).mean(axis=2)[:,region],np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['gfp_HD']],axis=2).std(axis=2)[:,region]/np.sqrt(subAvg['M']),color=colors.g())
    plot.lineError(ax[i,0],t,10*np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['HbT']],axis=2).mean(axis=2)[:,region],10*np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['HbT']],axis=2).std(axis=2)[:,region]/np.sqrt(subAvg['M']),color=colors.b())
    ax[i,0].set_xlim(-xlim,xlim)
    ax[i,0].axhline(y=0,ls=":",color="black")
    # ax[i,0].plot([t[0],t[-1]],[0,0],ls=":",color="black")
    [plot.remove_axes(ax[i,b],"LRTB") for b in range(4)]
    # plot.remove_axes(ax[i,0],"LRTB")
    
    bin_on = "prom"
    bins = [np.percentile(x,tuple(np.arange(0,100+100/n_bins,100/n_bins))) for x in subAvg[bin_on]]
    for b in range(n_bins):
        trials = [(x > bins[n][b]) & (x < bins[n][b+1]) for n,x in enumerate(subAvg[bin_on])]
        ax[i,1].plot(t,np.stack([x[:,:,trials[n]].mean(axis=2) for n,x in enumerate(subAvg['gfp_HD'])],axis=2).mean(axis=2)[:,region],
            color=cmp[b])
        ax[i,2].plot(t,np.stack([x[:,:,trials[n]].mean(axis=2) for n,x in enumerate(subAvg['HbT'])],axis=2).mean(axis=2)[:,region],
            color=cmp[b])
        ax[i,3].plot(t,100 * np.stack([x[:,trials[n]].mean(axis=1) for n,x in enumerate(subAvg['Pupil'])],axis=1).mean(axis=1),
            color=cmp[b])
        
    [ax[i,b+1].axvline(x=0,color='gray',ls=":") for b in range(3)]
    [ax[i,b+1].set_xlim(-xlim,xlim) for b in range(3)]

    if i == 0:
        titles = ["Ca2+","HbT"]
    else:
        titles = ["",""]
    gfp_img = np.stack([x.mean(axis=2) for x in subAvg['gfp_HD']],axis=2).max(axis=0)
    HbT_img = np.stack([x.mean(axis=2) for x in subAvg['HbT']],axis=2).max(axis=0)

    Ca_lags = [onsetDelay(a) for a in [(x.mean(axis=2)-x[0:30].mean(axis=(0,2)))[:,region] for x in subAvg['gfp_HD']]]
    HbT_lags = [onsetDelay(a) for a in [(x.mean(axis=2)-x[0:30].mean(axis=(0,2)))[:,region] for x in subAvg['HbT']]]
    delays.append([(HbT_lags[b]-Ca_lags[b])/10 for b in range(len(Ca_lags))])
    # delays.append(-t_xc[np.argmax(np.stack([x.mean(axis=2) for x in subAvg['gfp_HD']],axis=2)[:,4,:],axis=0)])

    stats = plot.scatter(ax[3,i],[x[:,region].max(axis=0)-x[0:30,region].mean(axis=0) for x in subAvg['gfp_HD']],[x[:,region].max(axis=0)-x[0:30,region].mean(axis=0) for x in subAvg['HbT']],
        alpha=0.3,
        xlabel="Ca2+ (% ∆F/F)",
        ylabel="HbT (µM)",
        cmp=mpl.colormaps["inferno"](np.linspace(0,0.8,7)),
        size=5)
    ax[3,i].set_title(f"r = {stats[2].mean():.2}")

# plot legends and scalebars
ax[0,0].legend(handles=[l for l in ax[0,0].lines][0:2],labels=["Ca2+","HbT"],frameon=False)
# ax[2,0].plot([5,9],[4,4],color="black")
# ax[2,0].plot([5,5],[4,9],color="black")

# plot bar plot
plot.bar(ax[3,3],delays,
    title="Hemodynamic delay",
    color=[colors.g(),colors.o(),colors.p()],
    linewidth=1,
    labels=["Rbp4","Calb","VIP"],
    ylabel="s",
    width=0.6,
    alpha=0.5,
    size=25)

ax[0,1].set_title("Ca2+")
ax[0,2].set_title("HbT")
ax[0,3].set_title("Pupil")

ax[0,0].set_ylabel("Rbp4")
ax[1,0].set_ylabel("Calb")
ax[2,0].set_ylabel("VIP")

ax[3,0].set_xlabel("Rbp4")
ax[3,1].set_xlabel("Calb")
ax[3,2].set_xlabel("VIP")

fig.savefig(fig_4 + "fig.svg", format="svg", bbox_inches="tight", dpi=600)
plt.show()

# %% plotting archived
n_bins = 3
cmp = mpl.colormaps['viridis'](np.linspace(0,1,n_bins+1))

region = 1

fig, ax = plt.subplots(7,4,figsize=(8,10.5))

t = np.arange(-10,10.1,0.1)
xlim = 10

t_xc = np.arange(-10,10.1,0.1)
spatial_r = []
delays = []
types = ["Rbp4","Calb","VIP"]

for i, subAvg in enumerate([peaks_Rbp4, peaks_Calb, peaks_VIP]):
    plot.lineError(ax[i+3,2],t,np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['gfp_HD']],axis=2).mean(axis=2)[:,region],np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['gfp_HD']],axis=2).std(axis=2)[:,region]/np.sqrt(subAvg['M']),color=colors.g())
    plot.lineError(ax[i+3,2],t,10*np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['HbT']],axis=2).mean(axis=2)[:,region],10*np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['HbT']],axis=2).std(axis=2)[:,region]/np.sqrt(subAvg['M']),color=colors.b())
    ax[i+3,2].set_xlim(-xlim,xlim)
    ax[i+3,2].plot([t[0],t[-1]],[0,0],ls=":",color="black")
    [plot.remove_axes(ax[i,b],"LRTB") for b in range(4)]
    plot.remove_axes(ax[i+3,2],"LRTB")

    # plot pupil
    # plot.lineError(ax[i,3],t,
    #     100 * np.stack([(x-x[0:30].mean(axis=0)).mean(axis=1) for x in subAvg['Pupil']],axis=1).mean(axis=1),
    #     100 * np.stack([(x-x[0:30].mean(axis=0)).mean(axis=1) for x in subAvg['Pupil']],axis=1).std(axis=1)/np.sqrt(subAvg['M']),
    #     color=colors.p())
    # ax[i,3].set_xlim(-xlim,xlim)
    # ax[i,3].axhline(y=0,ls=":",color="black")
    # ax[i,3].axvline(x=0,ls=":",color="black")
    # plot.remove_axes(ax[i,3],"LRTB")
    
    bin_on = "prom"
    bins = [np.percentile(x,tuple(np.arange(0,100+100/n_bins,100/n_bins))) for x in subAvg[bin_on]]
    for b in range(n_bins):
        trials = [(x > bins[n][b]) & (x < bins[n][b+1]) for n,x in enumerate(subAvg[bin_on])]
        ax[i,1].plot(t,np.stack([x[:,:,trials[n]].mean(axis=2) for n,x in enumerate(subAvg['gfp_HD'])],axis=2).mean(axis=2)[:,region],
                     color=cmp[b])
        ax[i,2].plot(t,np.stack([x[:,:,trials[n]].mean(axis=2) for n,x in enumerate(subAvg['HbT'])],axis=2).mean(axis=2)[:,region],
                             color=cmp[b])
        ax[i,3].plot(t,100 * np.stack([x[:,trials[n]].mean(axis=1) for n,x in enumerate(subAvg['Pupil'])],axis=1).mean(axis=1),
                                     color=cmp[b])
        
    [ax[i,b+1].axvline(x=0,color='gray',ls=":") for b in range(3)]
    [ax[i,b+1].set_xlim(-xlim,xlim) for b in range(3)]

    if i == 0:
        titles = ["Ca2+","HbT"]
    else:
        titles = ["",""]
    gfp_img = np.stack([x.mean(axis=2) for x in subAvg['gfp_HD']],axis=2).max(axis=0)
    HbT_img = np.stack([x.mean(axis=2) for x in subAvg['HbT']],axis=2).max(axis=0)

    plot.allenMap(ax[3+i,0],gfp_img.mean(axis=1),clim=None,clabel="∆F/F",ylabel=types[i],title=titles[0])
    plot.allenMap(ax[3+i,1],HbT_img.mean(axis=1),clim=None,clabel="µM",title=titles[1])
    spatial_r.append(IRF.corr(gfp_img,HbT_img,dim=0))
    Ca_lags = [onsetDelay(a) for a in [(x.mean(axis=2)-x[0:30].mean(axis=(0,2)))[:,4] for x in subAvg['gfp_HD']]]
    HbT_lags = [onsetDelay(a) for a in [(x.mean(axis=2)-x[0:30].mean(axis=(0,2)))[:,4] for x in subAvg['HbT']]]
    delays.append([(HbT_lags[b]-Ca_lags[b])/10 for b in range(len(Ca_lags))])
    # delays.append(-t_xc[np.argmax(np.stack([x.mean(axis=2) for x in subAvg['gfp_HD']],axis=2)[:,4,:],axis=0)])

    stats = plot.scatter(ax[6,i],[x[:,region].max(axis=0)-x[0:30,region].mean(axis=0) for x in subAvg['gfp_HD']],[x[:,region].max(axis=0)-x[0:30,region].mean(axis=0) for x in subAvg['HbT']],
            alpha=0.3,
            xlabel="Ca2+ (% ∆F/F)",
            ylabel="HbT (µM)",
            cmp=mpl.colormaps["inferno"](np.linspace(0,0.8,7)),
            size=5)

# plot peak detection
region = 4
signal = data_VIP[0]['parcells']['HbT'][:,region]
signal_lpf = bpf(signal, fr=[0,0.1], fs=10)

bounds = np.percentile(signal,(1,99))
peaks, prop = find_peaks(
    signal_lpf,
    prominence=(bounds[1]-bounds[0]) / 10,
    distance=1
)

t = np.arange(0,len(signal)/10,0.1)

ax[0,0].plot(t/60-15,signal,color=colors.b())
[ax[0,0].axvline(x=i/10/60-15,color='gray',ls=":") for i in peaks]
ax[0,0].set_xlim(0,5)
ax[0,0].set_xlabel("Time (min)")
ax[0,0].plot([4,4],[0,5],color="black")
plot.remove_axes(ax[0,0],"TLR")

# plot legends and scalebars
ax[3,2].legend(handles=[l for l in ax[3,2].lines][0:2],labels=["Ca2+","HbT"],frameon=False)
ax[5,2].plot([5,9],[4,4],color="black")
ax[5,2].plot([5,5],[4,9],color="black")

# plot bar plots
plot.bar(ax[3,3],spatial_r,
         title="Spatial Correlation",
         color=[colors.g(),colors.o(),colors.p()],
         linewidth=1,
         labels=["1","2","3"],
         ylabel="r",
         width=0.6,
         alpha=0.5,
         ylim=[0,1],
         size=25)

plot.bar(ax[4,3],delays,
         title="Hemodynamic delay",
         color=[colors.g(),colors.o(),colors.p()],
         linewidth=1,
         labels=["1","2","3"],
         ylabel="s",
         width=0.6,
         alpha=0.5,
         size=25)

ax[0,1].set_title("Ca2+")
ax[0,2].set_title("HbT")

fig.savefig(fig_4 + "fig.svg", format="svg", bbox_inches="tight", dpi=600)
plt.show()

# %% plot supplemental figure 1
def spc_analysis(data):
    log, order = create_log(data)

    Mice = np.unique(log.Mouse)

    subAvg = {}
    subAvg['Mice'] = Mice
    subAvg['M'] = len(Mice)

    subAvg['SPC_gfp_HD'] = []
    subAvg['SPC_HbT'] = []

    subAvg['SPG_gfp_HD'] = []
    subAvg['SPG_HbT'] = []

    subAvg['COH'] = []
    subAvg['PHI'] = []

    subAvg['SPG_low_gfp_HD'] = []
    subAvg['SPG_high_gfp_HD'] = []
    subAvg['SPG_low_HbT'] = []
    subAvg['SPG_high_HbT'] = []

    fr_steps = 50

    for mouse in Mice:
        session_idx = [i for i in range(len(log)) if log.Type[i] == "baseline" and log.Mouse[i] == mouse]

        SPC_gfp_HD = []
        SPC_HbT = []
        COH = []
        PHI = []

        SPG_gfp_HD = []
        SPG_HbT = []
        SPG_low_gfp_HD = []
        SPG_high_gfp_HD = []
        SPG_low_HbT = []
        SPG_high_HbT = []

        for i in session_idx:
            spc, subAvg['fr'] = spectra.mtspectrumc(data[i]['parcells']['gfp_HD'],Fs=10)
            SPC_gfp_HD.append(spc)
            spc, _ = spectra.mtspectrumc(data[i]['parcells']['raw_HbT'],Fs=10)
            SPC_HbT.append(spc)
            coh, phi, _ = spectra.coherencyc(data[i]['parcells']['gfp_HD'], 
                data[i]['parcells']['raw_HbT'],
                Fs=10)
            
            COH.append(coh)
            PHI.append(phi)

            pupil = data[i]['Pupil']
            pupil_bounds = np.percentile(pupil,(30,70))
            idx_high = pupil > pupil_bounds[1]
            idx_low = pupil < pupil_bounds[0]

            spg,_,subAvg['fr_spg'] = spectra.morlet(data[i]['parcells']['gfp_HD'],Fs=10,fwin=[0.01,5],steps=fr_steps)
            SPG_gfp_HD.append(spg)

            SPG_low_gfp_HD.append(spg[idx_low])
            SPG_high_gfp_HD.append(spg[idx_high])

            spg,_,subAvg['fr_spg'] = spectra.morlet(data[i]['parcells']['raw_HbT'],Fs=10,fwin=[0.01,5],steps=fr_steps)
            SPG_HbT.append(spg)

            SPG_low_HbT.append(spg[idx_low])
            SPG_high_HbT.append(spg[idx_high])

        subAvg['SPC_gfp_HD'].append(np.stack(SPC_gfp_HD,axis=2).mean(axis=2))
        subAvg['SPC_HbT'].append(np.stack(SPC_HbT,axis=2).mean(axis=2))
        subAvg['COH'].append(np.stack(COH,axis=2).mean(axis=2))
        subAvg['PHI'].append(np.stack(PHI,axis=2).mean(axis=2))
        subAvg['SPG_gfp_HD'].append(np.stack(SPG_gfp_HD,axis=3).mean(axis=3))
        subAvg['SPG_low_gfp_HD'].append(np.concatenate(SPG_low_gfp_HD,axis=0).mean(axis=0))
        # subAvg['SPG_low_gfp_HD'].append(np.stack(SPG_low_gfp_HD,axis=2))
        subAvg['SPG_high_gfp_HD'].append(np.concatenate(SPG_high_gfp_HD,axis=0).mean(axis=0))
        subAvg['SPG_low_HbT'].append(np.concatenate(SPG_low_HbT,axis=0).mean(axis=0))
        subAvg['SPG_high_HbT'].append(np.concatenate(SPG_high_HbT,axis=0).mean(axis=0))
        
    return subAvg

spc_Calb = spc_analysis(data_Calb)
spc_VIP = spc_analysis(data_VIP)
spc_Rbp4 = spc_analysis(data_Rbp4)

# %% plot

neuColors = [colors.g(),colors.o(),colors.p()]

fig, ax = plt.subplots(8,3,figsize=(8,13))
ax_r = []

fr_xlim = [0.01,5]
coh_maps = []
low_maps = []
high_maps = []
spc_maps = []

for i, subAvg in enumerate([spc_Rbp4, spc_Calb, spc_VIP]):
    fr = subAvg['fr']
    fr_spg = subAvg['fr_spg']
    # plot spectra
    plot.lineError(ax[0,0],fr,np.stack(subAvg['SPC_gfp_HD'],axis=2).mean(axis=(1,2)),np.stack(subAvg['SPC_gfp_HD'],axis=2).mean(axis=1).std(axis=1) / np.sqrt(subAvg['M']),
        yscale="log",
        xscale="log",
        xlim=fr_xlim,
        ylabel="PSD",
        color=neuColors[i])
    plot.lineError(ax[0,1],fr,np.stack(subAvg['SPC_HbT'],axis=2).mean(axis=(1,2)),np.stack(subAvg['SPC_HbT'],axis=2).mean(axis=1).std(axis=1) / np.sqrt(subAvg['M']),
        yscale="log",
        xscale="log",
        xlim=fr_xlim,
        ylabel="PSD",
        color=neuColors[i])

    # combined coherence plot
    plot.lineError(ax[0,2],fr,np.stack(subAvg['COH'],axis=2).mean(axis=(1,2)),np.stack(subAvg['COH'],axis=2).mean(axis=1).std(axis=1) / np.sqrt(subAvg['M']),
        xscale="log",
        xlim=fr_xlim,
        ylim=[0,1],
        ylabel="Coherence",
        color=neuColors[i])

    # coherence phi plots
    plot.lineError(ax[1,i],fr,np.stack(subAvg['COH'],axis=2).mean(axis=(1,2)),np.stack(subAvg['COH'],axis=2).mean(axis=1).std(axis=1) / np.sqrt(subAvg['M']),
        xscale="log",
        xlim=fr_xlim,
        ylim=[0,1],
        ylabel="Coherence" if i == 0 else "",
        color=neuColors[i])
    ax_r.append(ax[1,i].twinx())
    plot.lineError(ax_r[i],fr,np.stack(subAvg['PHI'],axis=2).mean(axis=(1,2)),np.stack(subAvg['PHI'],axis=2).mean(axis=1).std(axis=1) / np.sqrt(subAvg['M']),
        xscale="log",
        xlim=fr_xlim,
        ylim=[-np.pi,np.pi],
        ylabel="Phase (radians)" if i == 2 else "",
        color="black")
    ax_r[i].plot(fr_xlim,[0,0],color="black",ls=":")

    # coherence maps
    ax[2,i].remove()
    coh_maps_parent = ax[2,i].get_subplotspec().subgridspec(1,2)
    coh_maps.append([fig.add_subplot(coh_maps_parent[0,0]),fig.add_subplot(coh_maps_parent[0,1])])

    low_map = np.stack(subAvg['COH'],axis=2)[fr < 0.1].mean(axis=0)
    high_map = np.stack(subAvg['COH'],axis=2)[(fr > 0.1) & (fr < 0.5)].mean(axis=0)
    plot.allenMap(coh_maps[i][0],low_map.mean(axis=1),
        clim=[0,1],
        side="right",
        clabel="Coherence" if i == 2 else "",
        cmap="inferno",
        cbar=False,
        ylabel="< 0.1 Hz")
    plot.allenMap(coh_maps[i][1],high_map.mean(axis=1),
        clim=[0,1],
        side="right",
        clabel="Coherence" if i == 2 else "",
        cmap="inferno",
        cbar=True if i == 2 else False,
        ylabel="0.1-0.5 Hz")

    # arousal spectra
    plot.lineError(ax[3,i],fr_spg,
        fr_spg*np.stack(subAvg['SPG_low_gfp_HD'],axis=2).mean(axis=(1,2)),
        fr_spg*np.stack(subAvg['SPG_low_gfp_HD'],axis=2).mean(axis=1).std(axis=1) / np.sqrt(subAvg['M']),
        xscale="log",
        xlim=[0.01,5],
        color=colors.mb())
    plot.lineError(ax[3,i],fr_spg,
        fr_spg*np.stack(subAvg['SPG_high_gfp_HD'],axis=2).mean(axis=(1,2)),
        fr_spg*np.stack(subAvg['SPG_high_gfp_HD'],axis=2).mean(axis=1).std(axis=1) / np.sqrt(subAvg['M']),
        xscale="log",
        xlim=[0.01,5],
        color=colors.o())

    plot.lineError(ax[4,i],fr_spg,
        fr_spg*np.stack(subAvg['SPG_low_HbT'],axis=2).mean(axis=(1,2)),
        fr_spg*np.stack(subAvg['SPG_low_HbT'],axis=2).mean(axis=1).std(axis=1) / np.sqrt(subAvg['M']),
        xscale="log",
        xlim=[0.01,5],
        color=colors.mb())
    plot.lineError(ax[4,i],fr_spg,
        fr_spg*np.stack(subAvg['SPG_high_HbT'],axis=2).mean(axis=(1,2)),
        fr_spg*np.stack(subAvg['SPG_high_HbT'],axis=2).mean(axis=1).std(axis=1) / np.sqrt(subAvg['M']),
        xscale="log",
        xlim=[0.01,5],
        ylim=[0,0.075],
        color=colors.o())

    # low power maps
    ax[5,i].remove()
    low_maps_parent = ax[5,i].get_subplotspec().subgridspec(1,2)
    low_maps.append([fig.add_subplot(low_maps_parent[0,0]),fig.add_subplot(low_maps_parent[0,1])])

    gfp_map = (fr[:,None,None] * np.stack(subAvg['SPC_gfp_HD'],axis=2))[fr < 0.1].mean(axis=0)
    HbT_map = (fr[:,None,None] * np.stack(subAvg['SPC_HbT'],axis=2))[fr < 0.1].mean(axis=0)
    plot.allenMap(low_maps[i][0],gfp_map.mean(axis=1),
        side="right",
        cmap="inferno",
        cbar=True,
        ylabel="Low PSD" if i == 0 else "",
        title="Ca2+")
    plot.allenMap(low_maps[i][1],HbT_map.mean(axis=1),
        side="right",
        cmap="inferno",
        cbar=True,
        ylabel="",
        clim=[0,0.05],
        title="HbT")

    # high power maps
    ax[6,i].remove()
    high_maps_parent = ax[6,i].get_subplotspec().subgridspec(1,2)
    high_maps.append([fig.add_subplot(high_maps_parent[0,0]),fig.add_subplot(high_maps_parent[0,1])])

    gfp_map = (fr[:,None,None] * np.stack(subAvg['SPC_gfp_HD'],axis=2))[(fr > 0.1) & (fr < 5)].mean(axis=0)
    HbT_map = (fr[:,None,None] * np.stack(subAvg['SPC_HbT'],axis=2))[(fr > 0.1) & (fr < 0.5)].mean(axis=0)
    plot.allenMap(high_maps[i][0],gfp_map.mean(axis=1),
        side="right",
        cmap="inferno",
        cbar=True,
        ylabel="High PSD" if i == 0 else "")
    plot.allenMap(high_maps[i][1],HbT_map.mean(axis=1),
        side="right",
        cmap="inferno",
        cbar=True if i == 2 else False,
        clim=[0,0.075],
        ylabel="")
    
    # power ratio maps
    ax[7,i].remove()
    spc_maps_parent = ax[7,i].get_subplotspec().subgridspec(1,2)
    spc_maps.append([fig.add_subplot(spc_maps_parent[0,0]),fig.add_subplot(spc_maps_parent[0,1])])

    gfp_map = (fr[:,None,None] * np.stack(subAvg['SPC_gfp_HD'],axis=2))[(fr > 0.1) & (fr < 5)].mean(axis=0) / (fr[:,None,None] * np.stack(subAvg['SPC_gfp_HD'],axis=2))[fr < 0.1].mean(axis=0)
    HbT_map = (fr[:,None,None] * np.stack(subAvg['SPC_HbT'],axis=2))[(fr > 0.1) & (fr < 0.5)].mean(axis=0) / (fr[:,None,None] * np.stack(subAvg['SPC_HbT'],axis=2))[fr < 0.1].mean(axis=0)
    plot.allenMap(spc_maps[i][0],gfp_map.mean(axis=1),
        clim=[0,0.75],
        side="right",
        cmap="inferno",
        cbar=True if i == 2 else False,
        ylabel="High / Low PSD" if i == 0 else "")
    plot.allenMap(spc_maps[i][1],HbT_map.mean(axis=1),
        clim=[0,5],
        side="right",
        cmap="inferno",
        cbar=True if i == 2 else False,
        ylabel="")

[plot.remove_axes(ax[0,b],"TR") for b in range(3)]
[plot.remove_axes(ax[1,b],"T") for b in range(3)]
[plot.remove_axes(ax_r[b],"T") for b in range(3)]
[plot.remove_axes(ax[3,b],"TR") for b in range(3)]
[plot.remove_axes(ax[4,b],"TR") for b in range(3)]
[ax[0,b].set_xticks([0.01,0.1,1]) for b in range(3)]
[ax[0,b].set_xticklabels([0.01,0.1,1]) for b in range(3)]
[ax[1,b].set_xticks([0.01,0.1,1]) for b in range(3)]
[ax[1,b].set_xticklabels([0.01,0.1,1]) for b in range(3)]
[ax_r[b].set_yticks([-3.14,0,3.14]) for b in range(3)]
[ax_r[b].set_yticklabels(["-π",0,"π"]) for b in range(3)]
[ax[1,b].set_yticks([0,1]) for b in range(3)]
[ax[1,b].set_yticklabels([0,1]) for b in range(3)]
[ax[4,b].set_xticks([0.01,0.1,1]) for b in range(3)]
[ax[4,b].set_xticklabels([0.01,0.1,1]) for b in range(3)]
[ax[3,b].set_xticks([]) for b in range(3)]

ax[0,0].legend(handles=[l for l in ax[0,0].lines][0:3],labels=["Rbp4","Calb","VIP"],frameon=False)
ax[3,1].legend(handles=[l for l in ax[4,1].lines][0:3],labels=["low arousal","high arousal"],frameon=False)
[ax[4,b].set_xlabel("F (Hz)") for b in range(3)]
[ax[3+b,0].set_ylabel("PSD (normalized by F)") for b in range(2)]

# plot.prepare(fig)
fig.savefig(sfig_1 + "fig.svg", format="svg", bbox_inches="tight", dpi=600)
plt.show()

# %% region comparison

def reg_analysis(data):
    log, order = create_log(data)

    Mice = np.unique(log.Mouse)

    subAvg = {}
    subAvg['Mice'] = Mice
    subAvg['M'] = len(Mice)

    subAvg['SPC_gfp_HD'] = []
    
    for mouse in Mice:
        session_idx = [i for i in range(len(log)) if log.Type[i] == "baseline" and log.Mouse[i] == mouse]

        SPC_gfp_HD = []

        for i in session_idx:
            spc, subAvg['fr'] = spectra.mtspectrumc(data[i]['parcells']['gfp_HD'],Fs=10)
            SPC_gfp_HD.append(spc)
            spc, _ = spectra.mtspectrumc(data[i]['parcells']['raw_HbT'],Fs=10)
            SPC_HbT.append(spc)
            coh, phi, _ = spectra.coherencyc(data[i]['parcells']['gfp_HD'], 
                                            data[i]['parcells']['raw_HbT'],
                                            Fs=10)
            
            COH.append(coh)
            PHI.append(phi)

            pupil = data[i]['Pupil']
            pupil_bounds = np.percentile(pupil,(30,70))
            idx_high = pupil > pupil_bounds[1]
            idx_low = pupil < pupil_bounds[0]

            spg,_,subAvg['fr_spg'] = spectra.morlet(data[i]['parcells']['gfp_HD'],Fs=10,fwin=[0.01,5])
            SPG_gfp_HD.append(spg)

            SPG_low_gfp_HD.append(spg[idx_low])
            # SPG_low_gfp_HD.append(spg[idx_low].mean(axis=0))
            SPG_high_gfp_HD.append(spg[idx_high])

            spg,_,subAvg['fr_spg'] = spectra.morlet(data[i]['parcells']['raw_HbT'],Fs=10,fwin=[0.01,5])
            SPG_HbT.append(spg)

            SPG_low_HbT.append(spg[idx_low])
            SPG_high_HbT.append(spg[idx_high])

        subAvg['SPC_gfp_HD'].append(np.stack(SPC_gfp_HD,axis=2).mean(axis=2))
        subAvg['SPC_HbT'].append(np.stack(SPC_HbT,axis=2).mean(axis=2))
        subAvg['COH'].append(np.stack(COH,axis=2).mean(axis=2))
        subAvg['PHI'].append(np.stack(PHI,axis=2).mean(axis=2))
        subAvg['SPG_gfp_HD'].append(np.stack(SPG_gfp_HD,axis=3).mean(axis=3))
        subAvg['SPG_low_gfp_HD'].append(np.concatenate(SPG_low_gfp_HD,axis=0).mean(axis=0))
        # subAvg['SPG_low_gfp_HD'].append(np.stack(SPG_low_gfp_HD,axis=2))
        subAvg['SPG_high_gfp_HD'].append(np.concatenate(SPG_high_gfp_HD,axis=0).mean(axis=0))
        subAvg['SPG_low_HbT'].append(np.concatenate(SPG_low_HbT,axis=0).mean(axis=0))
        subAvg['SPG_high_HbT'].append(np.concatenate(SPG_high_HbT,axis=0).mean(axis=0))
        
    return subAvg

reg_Calb = reg_analysis(data_Calb)
reg_VIP = reg_analysis(data_VIP)
reg_Rbp4 = reg_analysis(data_Rbp4)

# %% movement analysis
def move_analysis(data):
    log, order = create_log(data)

    Mice = np.unique(log.Mouse)

    subAvg = {}
    subAvg['M'] = len(Mice)

    subAvg['gfp_HD'] = []
    subAvg['HbT'] = []
    subAvg['Acc'] = []
    subAvg['Pupil'] = []

    subAvg['prom'] = []
    subAvg['base'] = []

    subAvg['h_gfp_HD'] = []
    subAvg['h_HbT'] = []

    w = 100
    t_shift = np.arange(-w,w+1)
    
    for mouse in Mice:
        session_idx = [i for i in range(len(log)) if log.Type[i] == "baseline" and log.Mouse[i] == mouse]

        gfp_HD = []
        HbT = []
        Acc = []
        Pupil = []
        prom = []
        h_gfp_HD = []
        h_HbT = []

        for i in session_idx:
            region = 4
            signal = data[i]['Pupil']
            # signal = data[i]['parcells']['HbT'].mean(axis=1)
            signal = bpf(signal, fr=[0,0.1], fs=10)
            signal = np.diff(signal)
            
            bounds = np.percentile(signal,(1,99))
            peaks, prop = find_peaks(
                signal,
                prominence=(bounds[1] - bounds[0]) / 10,
                distance=15
            )

            # trim peaks
            N = data[i]['parcells']['gfp_HD'].shape[0]

            prom.append(np.array([x for i,x in enumerate(prop['prominences']) if peaks[i] >= w and peaks[i] < N-w]))
            peaks = [x for x in peaks if x >= w and x < N-w]

            h_gfp_HD.append(data[i]['parcells']['gfp_HD'][peaks,region])
            h_HbT.append(data[i]['parcells']['HbT'][peaks,region])

            for t in peaks:
                gfp_HD.append(data[i]['parcells_all']['gfp_HD'][t+t_shift,:])
                HbT.append(data[i]['parcells_all']['HbT'][t+t_shift,:])
                Pupil.append(data[i]['Pupil'][t+t_shift])
        
        subAvg['gfp_HD'].append(np.stack(gfp_HD,axis=2))
        subAvg['HbT'].append(np.stack(HbT,axis=2))
        subAvg['Pupil'].append(np.stack(Pupil,axis=1))
        subAvg['prom'].append(np.concat(prom))
        subAvg['h_gfp_HD'].append(np.concat(h_gfp_HD))
        subAvg['h_HbT'].append(np.concat(h_HbT))
        subAvg['base'].append(np.concat(h_gfp_HD)-np.concat(prom))
        
    return subAvg

move_Calb = move_analysis(data_Calb)
move_VIP = move_analysis(data_VIP)
move_Rbp4 = move_analysis(data_Rbp4)

def onsetDelay(data, baseline=30, bounds=(0.2,0.8)):
    data = data - data[0:baseline].mean(axis=0)
    data = data / data.max()
    N = np.argmax(data)
    data = data[N::-1]

    lag = []
    for i in range(2):
        idx = np.flatnonzero(data < bounds[i])[0]
        idx = idx - 1 + (data[idx-1] - bounds[i]) / (data[idx-1] - data[idx])
        lag.append(idx)

    lag = [N - lag[i] for i in range(2)]

    m = (bounds[1] - bounds[0]) / (lag[1] - lag[0])
    b = bounds[0] - lag[0] * m

    lag = -b / m

    return lag

# PLOTTING
n_bins = 3
cmp = mpl.colormaps['viridis'](np.linspace(0,1,n_bins+1))

region = 4

fig, ax = plt.subplots(4,4,figsize=(8,6))

t = np.arange(-10,10.1,0.1)
xlim = 10

t_xc = np.arange(-10,10.1,0.1)
spatial_r = []
delays = []
types = ["Rbp4","Calb","VIP"]

for i, subAvg in enumerate([move_Rbp4, move_Calb, move_VIP]):
    plot.lineError(ax[i,0],t,np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['gfp_HD']],axis=2).mean(axis=2)[:,region],np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['gfp_HD']],axis=2).std(axis=2)[:,region]/np.sqrt(subAvg['M']),color=colors.g())
    plot.lineError(ax[i,0],t,10*np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['HbT']],axis=2).mean(axis=2)[:,region],10*np.stack([x.mean(axis=2)-x[0:30].mean(axis=(0,2)) for x in subAvg['HbT']],axis=2).std(axis=2)[:,region]/np.sqrt(subAvg['M']),color=colors.b())
    ax[i,0].set_xlim(-xlim,xlim)
    ax[i,0].axhline(y=0,ls=":",color="black")
    # ax[i,0].plot([t[0],t[-1]],[0,0],ls=":",color="black")
    [plot.remove_axes(ax[i,b],"LRTB") for b in range(4)]
    # plot.remove_axes(ax[i,0],"LRTB")
    
    bin_on = "prom"
    bins = [np.percentile(x,tuple(np.arange(0,100+100/n_bins,100/n_bins))) for x in subAvg[bin_on]]
    for b in range(n_bins):
        trials = [(x > bins[n][b]) & (x < bins[n][b+1]) for n,x in enumerate(subAvg[bin_on])]
        ax[i,1].plot(t,np.stack([x[:,:,trials[n]].mean(axis=2) for n,x in enumerate(subAvg['gfp_HD'])],axis=2).mean(axis=2)[:,region],
            color=cmp[b])
        ax[i,2].plot(t,np.stack([x[:,:,trials[n]].mean(axis=2) for n,x in enumerate(subAvg['HbT'])],axis=2).mean(axis=2)[:,region],
            color=cmp[b])
        ax[i,3].plot(t,100 * np.stack([x[:,trials[n]].mean(axis=1) for n,x in enumerate(subAvg['Pupil'])],axis=1).mean(axis=1),
            color=cmp[b])
    
    [ax[i,b+1].axvline(x=0,color='gray',ls=":") for b in range(3)]
    [ax[i,b+1].set_xlim(-xlim,xlim) for b in range(3)]

    if i == 0:
        titles = ["Ca2+","HbT"]
    else:
        titles = ["",""]
    gfp_img = np.stack([x.mean(axis=2) for x in subAvg['gfp_HD']],axis=2).max(axis=0)
    HbT_img = np.stack([x.mean(axis=2) for x in subAvg['HbT']],axis=2).max(axis=0)

    # Ca_lags = [onsetDelay(a) for a in [(x.mean(axis=2)-x[0:30].mean(axis=(0,2)))[:,region] for x in subAvg['gfp_HD']]]
    # HbT_lags = [onsetDelay(a) for a in [(x.mean(axis=2)-x[0:30].mean(axis=(0,2)))[:,region] for x in subAvg['HbT']]]
    Ca_lags = [onsetDelay(a) for a in [(x.mean(axis=2)-x[0:30].mean(axis=(0,2)))[:,region] for x in subAvg['gfp_HD']]]
    HbT_lags = [onsetDelay(a) for a in [(x.mean(axis=2)-x[0:30].mean(axis=(0,2)))[:,region] for x in subAvg['HbT']]]
    delays.append([(HbT_lags[b]-Ca_lags[b])/10 for b in range(len(Ca_lags))])
    
    # delays.append(-t_xc[np.argmax(np.stack([x.mean(axis=2) for x in subAvg['gfp_HD']],axis=2)[:,4,:],axis=0)])

    stats = plot.scatter(ax[3,i],[x[:,region].max(axis=0)-x[0:30,region].mean(axis=0) for x in subAvg['gfp_HD']],[x[:,region].max(axis=0)-x[0:30,region].mean(axis=0) for x in subAvg['HbT']],
        alpha=0.3,
        xlabel="Ca2+ (% ∆F/F)",
        ylabel="HbT (µM)",
        cmp=mpl.colormaps["inferno"](np.linspace(0,0.8,7)),
        size=5)
    ax[3,i].set_title(f"r = {stats[2].mean():.2}")

# plot legends and scalebars
ax[0,0].legend(handles=[l for l in ax[0,0].lines][0:2],labels=["Ca2+","HbT"],frameon=False)
# ax[2,0].plot([5,9],[4,4],color="black")
# ax[2,0].plot([5,5],[4,9],color="black")

# plot bar plot
plot.bar(ax[3,3],delays,
    title="Hemodynamic delay",
    color=[colors.g(),colors.o(),colors.p()],
    linewidth=1,
    labels=["Rbp4","Calb","VIP"],
    ylabel="s",
    width=0.6,
    alpha=0.5,
    size=25)

ax[0,1].set_title("Ca2+")
ax[0,2].set_title("HbT")
ax[0,3].set_title("Pupil")

ax[0,0].set_ylabel("Rbp4")
ax[1,0].set_ylabel("Calb")
ax[2,0].set_ylabel("VIP")

ax[3,0].set_xlabel("Rbp4")
ax[3,1].set_xlabel("Calb")
ax[3,2].set_xlabel("VIP")

fig.savefig(fig_4 + "fig.svg", format="svg", bbox_inches="tight", dpi=600)
plt.show()

# %% movement detection test

signal = data_VIP[20]['Pupil']
signal_p = bpf(signal, fr=[0,0.1], fs=10)
signal_p = np.diff(signal_p)

bounds = np.percentile(signal_p,(1,99))
peaks, prop = find_peaks(
    signal_p,
    prominence=(bounds[1] - bounds[0]) / 10,
    distance=15
)

plt.plot(signal)
plt.scatter(peaks,signal[peaks],color="red")
plt.xlim([0,6000])
plt.show()

