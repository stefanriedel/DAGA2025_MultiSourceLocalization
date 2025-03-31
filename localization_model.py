import numpy as np
from Utility.auditory_cues import compute_auditory_cues_stationary, compute_auditory_cues_timevariant
from Utility.noise_psd import pink_noise
from os.path import dirname, join as pjoin
import scipy.signal as signal
from Utility.compute_block_gradients import compute_block_gradients
from Utility.compute_similarity_function import compute_similarity_function
from Utility.normal import normal
from Utility.scaled_sigmoid import scaled_sigmoid
from Utility.symmetrizeCircularHRTF import symmetrizeCircularHRIR
import scipy.signal as signal 

random_seed = 10 # To reproduce model results of paper, define fixed random seed

root_dir = dirname(__file__)
utility_dir = pjoin(root_dir, 'Utility')

# ~10 ms blocksize @ fs = 48 kHz with 50% overlap and rectangular window
blocksize = 512
hopsize = int(blocksize / 2)
fs = 48000
blocksize_sec = float(blocksize) / fs

# Load the 2D HRIR set of the KU100 dummy head
hrir_2D_dataset = np.load(file='./Utility/HRIR_CIRC360_48kHz.npy',
                          allow_pickle=True)
hrir_2D = hrir_2D_dataset[0]

# Symmetrize HRIR dataset for symmetric localization behaviour
hrir_2D = symmetrizeCircularHRIR(hrir_2D, 'Right')
hrir_l_2D = hrir_2D[:, 0, :]
hrir_r_2D = hrir_2D[:, 1, :]

# Load gammatone magnitude windows, precomputed using the 'pyfilterbank' library
# https://github.com/SiggiGue/pyfilterbank
filename = 'gammatone_erb_mag_windows_nfft_' + str(blocksize) + '_numbands_320.npy'
gammatone_mag_win = np.load(pjoin(utility_dir, filename))

# Every 8-th window for 1 ERB spacing
gammatone_mag_win = gammatone_mag_win[::8,:]
Nfft = int((gammatone_mag_win.shape[1] - 1) * 2)
num_bands = gammatone_mag_win.shape[0]
filename = 'gammatone_fc_numbands_320_fs_48000.npy'

f_c = np.load(pjoin(utility_dir, filename))
f_c = f_c[::8]

# Define head rotations for dynamic modeling
head_rotations = np.array([-15,0,15])
num_rotations = head_rotations.size

angular_res = 15
angular_range = 105 + 15
azi_angles = np.arange(-angular_range,angular_range+angular_res,angular_res)
num_directions = azi_angles.size

ITD_templates = np.zeros((num_bands, azi_angles.size))
ILD_templates = np.zeros((num_bands, azi_angles.size))
P_L_templates = np.zeros((num_bands, azi_angles.size))
P_R_templates = np.zeros((num_bands, azi_angles.size))

SpecGrad_L_templates = np.zeros((num_bands, azi_angles.size))
SpecGrad_R_templates = np.zeros((num_bands, azi_angles.size))

# Compute template cues from HRIRs
for azi, idx in zip(azi_angles, range(azi_angles.size)):
    hrir_l = hrir_l_2D[azi, :]
    hrir_r = hrir_r_2D[azi, :]

    IC, ITD, ILD, P_L, P_R = compute_auditory_cues_stationary(hrir_l, hrir_r, gammatone_mag_win, fs, blocksize)

    ITD_templates[:, idx] = ITD
    ILD_templates[:, idx] = ILD
    P_L_templates[:, idx] = P_L
    P_R_templates[:, idx] = P_R
    SpecGrad_L_templates[:, idx], SpecGrad_R_templates[:, idx]  = compute_block_gradients(P_L, P_R)

# Compute stimulus (target) cues from simulated binaural signals 
stim_length = int(2 * fs)
num_blocks = int(np.floor(stim_length / hopsize) - 1)

target_angles = np.arange(-105, 105+15, 15)

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
result_idcs = np.arange(np.where(azi_angles == -105)[0][0], np.where(azi_angles == 105)[0][0] + 1)

all_p_itd = np.zeros((num_conditions, num_directions, num_rotations))
all_p_ild = np.zeros((num_conditions, num_directions, num_rotations))
all_p_spec = np.zeros((num_conditions, num_directions, num_rotations))

for rotation, rot_idx in zip(head_rotations, range(num_rotations)):
    print('Calculating Rotation: ' + str(rotation))

    random_seed = random_seed + rot_idx
    rng = np.random.default_rng(random_seed)

    for idcs, cond_idx in zip(target_idcs, range(num_conditions)):
        y_L = np.zeros(int(stim_length + hrir_l_2D.shape[1] - 1))
        y_R = np.zeros(int(stim_length + hrir_l_2D.shape[1] - 1))

        angle_idcs = np.asarray(idcs) == 1
        angles = target_angles[angle_idcs]
        angles = np.asarray(angles)

        angles -= rotation

        for idx in angles:
            audio_buffer = pink_noise(stim_length, rng)
            y_L += signal.fftconvolve(audio_buffer, hrir_l_2D[idx, :])
            y_R += signal.fftconvolve(audio_buffer, hrir_r_2D[idx, :])

        IC, ITD, ILD, P_L, P_R = compute_auditory_cues_timevariant(y_L, y_R, gammatone_mag_win, fs, blocksize, 
                                                                   hopsize=hopsize, num_blocks=num_blocks)
                
        # Interaural time differences (ITDs)
        p_itd = np.ones((num_directions, num_bands, num_blocks))
        low_band_itd = np.where(f_c >= 500)[0][0] 
        high_band_itd =  np.where(f_c >= 1400)[0][0] 

        for block in range(num_blocks):
            for band in range(low_band_itd, high_band_itd):
                IC_stimulus = IC[block,band]
                sigma_ITD = scaled_sigmoid(1 - IC_stimulus, x0=0.5, k=10, y_min=0.00005, y_max=0.001)
                p_itd[:, band, block] = normal(ITD[block,band], ITD_templates[band, :], sigma_ITD)
                p_itd[:, band, block] /= np.sum(p_itd[:, band, block])

        # Integration across time and frequency
        avg_p_itd = np.mean(np.mean(p_itd[:,low_band_itd:high_band_itd,:], axis=-1), axis=-1)
        rot_avg_p_itd = np.roll(avg_p_itd, shift=int(rotation/angular_res))
        all_p_itd[cond_idx, :, rot_idx] = rot_avg_p_itd

        # Interaural level differences (ILDs)
        p_ild = np.ones((num_directions, num_bands, num_blocks))
        low_band_ild = np.where(f_c >= 3200)[0][0] 
        high_band_ild = np.where(f_c >= 8000)[0][0] 

        for block in range(num_blocks):
            for band in range(low_band_ild, high_band_ild):
                IC_stimulus = IC[block,band]
                sigma_ILD = scaled_sigmoid(1 - IC_stimulus, x0=0.5, k=10, y_min=1, y_max=10)
                p_ild[:, band, block] = normal(ILD[block,band], ILD_templates[band, :], sigma_ILD)
                p_ild[:, band, block] /= np.sum(p_ild[:, band, block]) 

        # Integration across time and frequency
        avg_p_ild = np.mean(np.mean(p_ild[:,low_band_ild:high_band_ild,:], axis=-1), axis=-1)
        rot_avg_p_ild = np.roll(avg_p_ild, shift=int(rotation/angular_res))
        all_p_ild[cond_idx, :, rot_idx] = rot_avg_p_ild

        # Monaural spectral cues (SC)
        SpecGrad_L = np.zeros((num_bands, num_blocks))
        SpecGrad_R = np.zeros((num_bands, num_blocks))

        Sim_Func_L = np.zeros((num_directions, num_blocks))
        Sim_Func_R = np.zeros((num_directions, num_blocks))

        prob_spec_cues_l = np.zeros((num_directions, num_blocks))
        prob_spec_cues_r = np.zeros((num_directions, num_blocks))

        avg_p_speccues = np.zeros((num_directions, num_blocks))

        low_band = np.where(f_c >= 3200)[0][0] 
        high_band = np.where(f_c >= 18000)[0][0]
        eps = 1e-6

        for block in range(num_blocks):
            SpecGrad_L[:,block], SpecGrad_R[:,block] = compute_block_gradients(P_X_L=P_L[block,:], P_X_R=P_R[block,:])

            Sim_Func_L[:,block], Sim_Func_R[:,block] = compute_similarity_function(SpecGrad_L[:,block], SpecGrad_R[:,block], 
                                                                                    SpecGrad_L_templates, SpecGrad_R_templates, 
                                                                                    f_c, low_band, high_band)

            prob_spec_cues_l[:,block] = Sim_Func_L[:,block] / (np.sum(Sim_Func_L[:,block]) + eps)         
            prob_spec_cues_r[:,block] = Sim_Func_R[:,block] / (np.sum(Sim_Func_R[:,block]) + eps)

            avg_p_speccues[:,block] = (prob_spec_cues_l[:,block] + prob_spec_cues_r[:,block])
            avg_p_speccues[:,block] /= np.sum(avg_p_speccues[:,block] + eps)

        avg_p_speccues = np.mean(avg_p_speccues, axis=-1)
        rot_avg_p_speccues = np.roll(avg_p_speccues, shift=int(rotation/angular_res))
        all_p_spec[cond_idx, :, rot_idx] = rot_avg_p_speccues

np.save('./ModelOutput/all_p_itd', arr=all_p_itd[:,result_idcs,:], allow_pickle=True)
np.save('./ModelOutput/all_p_ild', arr=all_p_ild[:,result_idcs,:], allow_pickle=True)
np.save('./ModelOutput/all_p_spec', arr=all_p_spec[:,result_idcs,:], allow_pickle=True)






