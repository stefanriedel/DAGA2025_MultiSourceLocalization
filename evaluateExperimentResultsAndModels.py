import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
import os
from datetime import datetime
import json
import scipy.stats as stats

PLOT_MODEL_DATA = True

# Stimuli: Target indices from -105 to -105 degrees
target_idcs = [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
               [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
               [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
               [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
               [0,0,0,0,0,1,0,1,0,1,0,0,0,0,0],
               [0,0,0,0,1,0,0,1,0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
               [0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
               [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0]]
num_conditions = len(target_idcs)
num_directions = 15

root_dir = dirname(__file__)
results_dir = pjoin(root_dir, 'ExpResults')
results_dir_static = pjoin(results_dir, 'static')
results_dir_dynamic = pjoin(results_dir, 'dynamic')

# Load results
file_list_static = os.listdir(results_dir_static)
if '.DS_Store' in file_list_static:
    file_list_static.remove('.DS_Store')
file_list_dynamic = os.listdir(results_dir_dynamic)
if '.DS_Store' in file_list_static:
    file_list_dynamic.remove('.DS_Store')

num_participants = len(file_list_dynamic)

ages = []
genders = []
exp_times_min_static = []
exp_times_min_dynamic = []

results_static_broadband = np.zeros((num_participants, num_conditions, num_directions))
results_static_lowpass = np.zeros((num_participants, num_conditions, num_directions))
results_static_highpass = np.zeros((num_participants, num_conditions, num_directions))

results_dynamic_broadband = np.zeros((num_participants, num_conditions, num_directions))
results_dynamic_lowpass = np.zeros((num_participants, num_conditions, num_directions))
results_dynamic_highpass = np.zeros((num_participants, num_conditions, num_directions))

for subj in range(num_participants):
    # Read static results
    f_location = pjoin(results_dir_static, file_list_static[subj])
    f = open(f_location)
    json_file = json.load(f)

    start_time = datetime.strptime(json_file['StartTime'], "%d %b %Y %H:%M:%S")
    end_time = datetime.strptime(json_file['EndTime'], "%d %b %Y %H:%M:%S")

    exp_time_seconds = (end_time - start_time)
    exp_times_min_static.append(int(exp_time_seconds.total_seconds() / 60.0))

    data = json_file['Results']['Parts']

    for cond in range(num_conditions):
        results_static_broadband[subj,cond,:] = data[0]['Trials'][cond]['Ratings'][1:]
        results_static_lowpass[subj,cond,:] = data[1]['Trials'][cond]['Ratings'][1:]
        results_static_highpass[subj,cond,:] = data[2]['Trials'][cond]['Ratings'][1:]

    # Read dynamic results
    f_location = pjoin(results_dir_dynamic, file_list_dynamic[subj])
    f = open(f_location)
    json_file = json.load(f)

    start_time = datetime.strptime(json_file['StartTime'], "%d %b %Y %H:%M:%S")
    end_time = datetime.strptime(json_file['EndTime'], "%d %b %Y %H:%M:%S")

    exp_time_seconds = (end_time - start_time)
    exp_times_min_dynamic.append(int(exp_time_seconds.total_seconds() / 60.0))

    data = json_file['Results']['Parts']

    for cond in range(num_conditions):
        results_dynamic_broadband[subj,cond,:] = data[0]['Trials'][cond]['Ratings'][1:]
        results_dynamic_lowpass[subj,cond,:] = data[1]['Trials'][cond]['Ratings'][1:]
        results_dynamic_highpass[subj,cond,:] = data[2]['Trials'][cond]['Ratings'][1:]

    ages.append(json_file['SubjectInformation']['Age'])
    genders.append(json_file['SubjectInformation']['Gender'])

# Load model results
all_p_itd = np.load('./ModelOutput/all_p_itd.npy', allow_pickle=True)
all_p_ild = np.load('./ModelOutput/all_p_ild.npy', allow_pickle=True)
all_p_spec = np.load('./ModelOutput/all_p_spec.npy', allow_pickle=True)

num_rotations = all_p_itd.shape[2]
frontal_head_idx = int(np.floor(num_rotations / 2))

title_list = ['Broadband (Static)', 'Low-pass (Static)', 'High-pass (Static)',
              'Broadband (Dynamic)', 'Low-pass (Dynamic)', 'High-pass (Dynamic)']
results_list = [results_static_broadband, results_static_lowpass, results_static_highpass,
                results_dynamic_broadband, results_dynamic_lowpass, results_dynamic_highpass]
savename_list = ['BB_Static', 'LP_Static', 'HP_Static', 'BB_Dynamic', 'LP_Dynamic', 'HP_Dynamic']

conditions_to_plot = np.array([0,1,2,3,5,11])

LEV_model = np.zeros((len(title_list), conditions_to_plot.size))
LEV_data = np.zeros((len(title_list), conditions_to_plot.size))

def LEV(p):
    return 1 - (np.max(p) - np.mean(p))

def total_variation_distance(p, q):
    """
    Computes the Total Variation Distance between two discrete probability distributions.
    p, q: Arrays representing discrete probability distributions.
    Returns a value in [0,1].
    """
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))

for idx in range(6):
    size = 4
    fig, axs = plt.subplots(ncols=1, nrows=conditions_to_plot.size, sharex=False, sharey=False, 
                            figsize=(1.25*size,2*size*(conditions_to_plot.size/num_conditions)), 
                            gridspec_kw={'hspace': 0.75, 'wspace': 0.0})
    xlabels = np.arange(-105, 105+15, step=15)
    for cond, ax_idx in zip(conditions_to_plot, range(conditions_to_plot.size)):#range(num_conditions):
        count =  np.sum(results_list[idx][:,cond,:], axis=0) 
        count = count / np.max(count)
        axs[ax_idx].bar(xlabels, count, width=12.5, facecolor='gray', edgecolor='k')
        axs[ax_idx].plot(xlabels[np.array(target_idcs[cond], dtype=bool)], 
                          [0.1] * np.asarray(target_idcs[cond]).sum(), 
                          marker='s', ls='', color='k')
        axs[ax_idx].set_ylim(0,1)
        axs[ax_idx].set_xlim(-120,120)
        axs[ax_idx].set_yticks([0,1], ['0', '1'], fontsize=9)
        axs[ax_idx].set_xticks(xlabels, labels=[str(x) for x in xlabels], fontsize=9)
        if cond == 11:
            axs[ax_idx].set_xlabel('Azimuth in Deg.')

        LEV_data[idx, ax_idx] = LEV(count)
        
        if PLOT_MODEL_DATA:
            lw = 2.5
            ls = '-'
            if savename_list[idx] == 'LP_Static':
                p = all_p_itd[cond,:,frontal_head_idx] / np.max(all_p_itd[cond,:,frontal_head_idx])
                axs[ax_idx].plot(xlabels, p, ls=ls, lw=lw, marker='', color='cornflowerblue')

                TVD = total_variation_distance(count / np.sum(count), p / np.sum(p))
                axs[ax_idx].text(125,0.7,s=r'$\text{TVD}_\text{ITD} = $' + str(round(TVD, 2)), fontsize=8)
                axs[ax_idx].text(125,0.2,s=r'$\text{LEV}_\text{ITD} = $' + str(round(LEV(p), 2)), fontsize=8)
                LEV_model[idx,ax_idx] = LEV(p)
            if savename_list[idx] == 'LP_Dynamic':
                p = np.ones(num_directions)
                for rot in range(num_rotations):
                    p *= all_p_itd[cond,:,rot]
                p /= np.max(p)
                axs[ax_idx].plot(xlabels, p, ls=ls, lw=lw, marker='', color='cornflowerblue', label='ITD')

                if ax_idx == 0:
                    axs[ax_idx].legend(loc='upper right', bbox_to_anchor=(1.015, 1.18), borderpad=0.2, framealpha=1.0, handletextpad=0.4)

                TVD = total_variation_distance(count / np.sum(count), p / np.sum(p))
                axs[ax_idx].text(125,0.7,s=r'$\text{TVD}_\text{ITD} = $' + str(round(TVD, 2)), fontsize=8)
                axs[ax_idx].text(125,0.2,s=r'$\text{LEV}_\text{ITD} = $' + str(round(LEV(p), 2)), fontsize=8)
                LEV_model[idx,ax_idx] = LEV(p)
            if savename_list[idx] == 'HP_Static':
                p = all_p_ild[cond,:,frontal_head_idx] / np.max(all_p_ild[cond,:, frontal_head_idx])
                axs[ax_idx].plot(xlabels, p, ls=ls, lw=lw, marker='', color='tab:olive')

                TVD_ild = total_variation_distance(count / np.sum(count), p / np.sum(p))
                LEV_ild = LEV(p)

                p = all_p_spec[cond,:,frontal_head_idx] / np.max(all_p_spec[cond,:, frontal_head_idx])
                axs[ax_idx].plot(xlabels, p, ls=ls, lw=lw, marker='', color='tab:pink')

                TVD_sc = total_variation_distance(count / np.sum(count), p / np.sum(p))
                LEV_sc = LEV(p)

                axs[ax_idx].text(125,0.9,s=r'$\text{TVD}_\text{ILD} = $' + str(round(TVD_ild, 2)), fontsize=8)
                axs[ax_idx].text(125,0.6,s=r'$\text{TVD}_\text{SC} = $' + str(round(TVD_sc, 2)), fontsize=8)

                axs[ax_idx].text(125,0.2,s=r'$\text{LEV}_\text{ILD} = $' + str(round(LEV_ild, 2)), fontsize=8)
                axs[ax_idx].text(125,-0.1,s=r'$\text{LEV}_\text{SC} = $' + str(round(LEV_sc, 2)), fontsize=8)
                LEV_model[idx,ax_idx] = LEV_sc
            if savename_list[idx] == 'HP_Dynamic':
                p = np.ones(num_directions)
                for rot in range(num_rotations):
                    p *= all_p_ild[cond,:,rot]
                p /= np.max(p)
                axs[ax_idx].plot(xlabels, p, ls=ls, lw=lw, marker='', color='tab:olive', label='ILD')

                TVD_ild = total_variation_distance(count / np.sum(count), p / np.sum(p))
                LEV_ild = LEV(p)

                p = np.ones(num_directions)
                for rot in range(num_rotations):
                    p *= all_p_spec[cond,:,rot]
                p /= np.max(p)
                axs[ax_idx].plot(xlabels, p, ls=ls, lw=lw, marker='', color='tab:pink', label='SC')

                TVD_sc = total_variation_distance(count / np.sum(count), p / np.sum(p))
                LEV_sc = LEV(p)

                if ax_idx == 0:
                    axs[ax_idx].legend(loc='upper right', bbox_to_anchor=(1.015, 1.18), borderpad=0.2, framealpha=1.0, ncol=2, columnspacing=1.0, handletextpad=0.4)

                axs[ax_idx].text(125,0.9,s=r'$\text{TVD}_\text{ILD} = $' + str(round(TVD_ild, 2)), fontsize=8)
                axs[ax_idx].text(125,0.6,s=r'$\text{TVD}_\text{SC} = $' + str(round(TVD_sc, 2)), fontsize=8)

                axs[ax_idx].text(125,0.2,s=r'$\text{LEV}_\text{ILD} = $' + str(round(LEV_ild, 2)), fontsize=8)
                axs[ax_idx].text(125,-0.1,s=r'$\text{LEV}_\text{SC} = $' + str(round(LEV_sc, 2)), fontsize=8)
                LEV_model[idx,ax_idx] = LEV_sc
            if savename_list[idx] == 'BB_Static':
                p = all_p_itd[cond,:,frontal_head_idx] + all_p_spec[cond,:,frontal_head_idx]
                p /= np.max(p) 
                axs[ax_idx].plot(xlabels, p, ls=ls, lw=lw, marker='', color='tab:purple')

                TVD = total_variation_distance(count / np.sum(count), p / np.sum(p))
                axs[ax_idx].text(125,0.7,s=r'$\text{TVD}_\text{ITD+SC} = $' + str(round(TVD, 2)), fontsize=8)
                axs[ax_idx].text(125,0.2,s=r'$\text{LEV}_\text{ITD+SC} = $' + str(round(LEV(p), 2)), fontsize=8)
                LEV_model[idx,ax_idx] = LEV(p)
            if savename_list[idx] == 'BB_Dynamic':
                p = np.ones(num_directions)
                for rot in range(num_rotations):
                    p *= all_p_itd[cond,:,rot] + all_p_spec[cond,:,rot]
                p /= np.max(p)
                axs[ax_idx].plot(xlabels, p, ls=ls,lw=lw, marker='', color='tab:purple', label='ITD + SC')
                if ax_idx == 0:
                    axs[ax_idx].legend(loc='upper right', bbox_to_anchor=(1.015, 1.18), borderpad=0.2, framealpha=1.0, handletextpad=0.4)
                
                TVD = total_variation_distance(count / np.sum(count), p / np.sum(p))
                axs[ax_idx].text(125,0.7,s=r'$\text{TVD}_\text{ITD+SC} = $' + str(round(TVD, 2)), fontsize=8)
                axs[ax_idx].text(125,0.2,s=r'$\text{LEV}_\text{ITD+SC} = $' + str(round(LEV(p), 2)), fontsize=8)
                LEV_model[idx,ax_idx] = LEV(p)

    axs[0].set_title(title_list[idx])
    plt.savefig('./Figures/' + savename_list[idx] + '.eps', bbox_inches='tight', dpi=300)

lp_data = np.concatenate((LEV_data[1,:],LEV_data[4,:]))
lp_model = np.concatenate((LEV_model[1,:],LEV_model[4,:]))
r_lp, p_lp = stats.pearsonr(lp_data, lp_model)

bb_data = np.concatenate((LEV_data[0,:],LEV_data[3,:]))
bb_model = np.concatenate((LEV_model[0,:],LEV_model[3,:]))
r_bb, p_bb = stats.pearsonr(bb_data, bb_model)

hp_data = np.concatenate((LEV_data[2,:],LEV_data[5,:]))
hp_model = np.concatenate((LEV_model[0,:],LEV_model[3,:]))
r_hp, p_hp = stats.pearsonr(hp_data, hp_model)

all_data = np.concatenate((lp_data, bb_data, hp_data))
all_model = np.concatenate((lp_model, bb_model, hp_model))
r_all, p_all = stats.pearsonr(all_data, all_model)

print('done')


