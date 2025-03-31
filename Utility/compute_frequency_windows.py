import numpy as np
from pyfilterbank import GammatoneFilterbank as GT


def compute_freq_windows(blocksize, sampling_rate, density=1):
    # Prepare filterbank
    gammatone_filterbank = GT(samplerate=sampling_rate, normfreq=1000.0, startband=-14, endband=26, order=4, bandwidth_factor=1,  density=density)
    # Get list of frequency responses
    gt_list = gammatone_filterbank.freqz(nfft=blocksize, plotfun=None)
    num_bands = len(gt_list)
    # Get center frequency for plots
    window_center_freq = gammatone_filterbank.centerfrequencies
    # Init container for frequency responses and store them
    freq_windows = np.zeros((num_bands, int(blocksize/2+1)), dtype=complex)
    for b in range(num_bands):
        freq_windows[b,:] = gt_list[b][0]

    return freq_windows, window_center_freq