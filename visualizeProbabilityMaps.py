import matplotlib.pyplot as plt
import numpy as np

rot_idx = 1 # frontal head
cond_idx = 3 # e.g. 3 equals +-45 deg.

p_itd_dict = np.load('./ModelOutput/p_itd_dict.npy', allow_pickle=True)
p_ild_dict = np.load('./ModelOutput/p_ild_dict.npy', allow_pickle=True)
p_sc_dict = np.load('./ModelOutput/p_sc_dict.npy', allow_pickle=True)

# rot x cond x dirs x bands x blocks:
p_itd_full_data = p_itd_dict.item().get('p_itd_full_data')
# rot x cond x dirs x bands x blocks:
p_ild_full_data = p_ild_dict.item().get('p_ild_full_data')
# rot x cond x dirs x bands x blocks:
p_sc_full_data = p_sc_dict.item().get('p_sc_full_data')

angular_range = 105
angular_res = 15
azi_angles = np.arange(-angular_range,angular_range+angular_res,angular_res)
num_directions = azi_angles.size

p_itd_bands = p_itd_full_data[rot_idx,cond_idx,:,:,:]
p_itd_instant = np.mean(p_itd_bands, axis=1)

p_ild_bands = p_ild_full_data[rot_idx,cond_idx,:,:,:]
p_ild_instant = np.mean(p_ild_bands, axis=1)

p_sc_instant = p_sc_full_data[rot_idx,cond_idx,:,:]

p_s = [p_itd_instant, p_ild_instant, p_sc_instant]
titles = [r'$P(\alpha | \text{ITD})$', r'$P(\alpha | \text{ILD})$', r'$P(\alpha | \text{SC})$']
savenames =  ['P_ITD', 'P_ILD', 'P_SC']

num_blocks = 40
for p, title, savename in zip(p_s, titles, savenames):
    fig, axs = plt.subplots(nrows=num_blocks,ncols=1,figsize=(4,4),gridspec_kw={'hspace': -0.8})
    for block, ax_idx in zip(range(num_blocks), np.arange(num_blocks-1,-1,-1)):
        # plotting the distribution
        axs[ax_idx].plot(azi_angles, p[:,block], color="k",lw=2)
        axs[ax_idx].fill_between(azi_angles, p[:,block], alpha=1,color='gray')

        # setting uniform x and y lims
        axs[ax_idx].set_xlim(-angular_range, angular_range)
        axs[ax_idx].set_ylim(0,0.75)

        # make background transparent
        rect = axs[ax_idx].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        axs[ax_idx].set_yticklabels([])

        if ax_idx == num_blocks-1:
            axs[ax_idx].set_xlabel("Azimuth in Deg.")
            axs[ax_idx].set_xticks(np.array([-90,-45,0,45,90]))
            axs[ax_idx].set_yticks([])
        else:
            axs[ax_idx].set_xticklabels([])
            axs[ax_idx].set_xticks([])
            axs[ax_idx].set_yticks([])

        spines = ["top","right","left","bottom"]
        for s in spines:
            axs[ax_idx].spines[s].set_visible(False)

        if ax_idx == 2:
            axs[ax_idx].set_title(title)
        if ax_idx == num_blocks / 2:
            axs[ax_idx].set_ylabel('Time')

    plt.savefig('./Figures/' + savename + '.png', dpi=300, bbox_inches='tight')


