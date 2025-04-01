import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import scipy.signal as signal

def compute_auditory_cues_timevariant(x_L, x_R, gammatone_mag_win, fs, blocksize, hopsize,
                          num_blocks):
    """ Compute per block: interaural coherence (IC), interaural time differences (ITD),
    interaural level difference (ILD), and monaural spectra.
    Args:
        x_L (ndarray): left-ear input signal in time-domain
        x_R (ndarray): right-ear input signal in time-domain
        gammatone_mag_win (ndarray): precomputed zero-phase magnitude bandpass windows
        fs (float): sampling rate in Hz
        blocksize (int): blocksize in samples for computation of FFT spectrum
        hopsize (int): hopsize in samples for computation of FFT spectrum
        num_blocks (int): number of blocks 

    Returns:
        ndarray: IC [t, num_bands]
        ndarray: ITD [t, num_bands]
        ndarray: ILD [t, num_bands]
        ndarray: P_L [t, num_bands]
        ndarray: P_R [t, num_bands]
    """
    eps = 1e-6
    num_bands = gammatone_mag_win.shape[0]
    tau_limit = 0.001
    tau_range = np.arange(int(-fs * tau_limit), int(fs * tau_limit))

    IC = np.zeros((num_blocks, num_bands))
    ILD = np.zeros((num_blocks, num_bands))
    ITD = np.zeros((num_blocks, num_bands))
    P_L = np.zeros((num_blocks, num_bands))
    P_R = np.zeros((num_blocks, num_bands))

    for t in range(num_blocks):
        x_L_block = x_L[t * hopsize:(t * hopsize + blocksize)]
        x_R_block = x_R[t * hopsize:(t * hopsize + blocksize)]

        X_L = np.fft.rfft(x_L_block)
        X_R = np.fft.rfft(x_R_block)

        cross_spectrum = np.conj(X_L) * X_R
        auto_spectrum_L = np.conj(X_L) * X_L
        auto_spectrum_R = np.conj(X_R) * X_R

        for b in range(num_bands):
            window = gammatone_mag_win[b, :]**2
            cross_spec_w = cross_spectrum * window
            auto_spec_l_w = auto_spectrum_L * window
            auto_spec_r_w = auto_spectrum_R * window

            cross_correlation = np.real(np.fft.irfft(cross_spec_w))
            P_l = np.fft.irfft(auto_spec_l_w)[0]
            P_r = np.fft.irfft(auto_spec_r_w)[0]

            IC[t, b] = (np.max(np.abs(cross_correlation[tau_range])) +
                        eps) / np.sqrt((P_l + eps) * (P_r + eps))
            ILD[t, b] = 10 * np.log10((P_l + eps) / (P_r + eps))
            ITD[t,
                b] = (np.argmax(cross_correlation[tau_range]) / fs) - tau_limit
            P_L[t, b] = P_l
            P_R[t, b] = P_r

    return IC, ITD, ILD, P_L, P_R

def compute_auditory_cues_stationary(x_L, x_R, gammatone_mag_win, fs, blocksize):
    """ Compute for one block: interaural coherence (IC), interaural time differences (ITD),
    interaural level difference (ILD), and monaural spectra.
    Args:
        x_L (ndarray): left-ear input signal in time-domain
        x_R (ndarray): right-ear input signal in time-domain
        gammatone_mag_win (ndarray): precomputed zero-phase magnitude bandpass windows
        fs (float): sampling rate in Hz
        blocksize (int): blocksize in samples for computation of FFT spectrum
    Returns:
        ndarray: IC [num_bands]
        ndarray: ITD [num_bands]
        ndarray: ILD [num_bands]
        ndarray: P_L [num_bands]
        ndarray: P_R [num_bands]
    """
    eps = 1e-6
    num_bands = gammatone_mag_win.shape[0]
    tau_limit = 0.001
    tau_range = np.arange(int(-fs * tau_limit), int(fs * tau_limit))

    IC = np.zeros(num_bands)
    ILD = np.zeros(num_bands)
    ITD = np.zeros(num_bands)
    P_L = np.zeros(num_bands)
    P_R = np.zeros(num_bands)

    x_L_block = x_L # left hrir
    x_R_block = x_R # right hrir

    X_L = np.fft.rfft(x_L_block, n=blocksize)
    X_R = np.fft.rfft(x_R_block, n=blocksize)

    cross_spectrum = np.conj(X_L) * X_R
    auto_spectrum_L = np.conj(X_L) * X_L
    auto_spectrum_R = np.conj(X_R) * X_R

    for b in range(num_bands):
        window = gammatone_mag_win[b, :]**2
        cross_spec_w = cross_spectrum * window
        auto_spec_l_w = auto_spectrum_L * window
        auto_spec_r_w = auto_spectrum_R * window

        cross_correlation = np.real(np.fft.irfft(cross_spec_w))
        P_l = np.fft.irfft(auto_spec_l_w)[0]
        P_r = np.fft.irfft(auto_spec_r_w)[0]

        IC[b] = (np.max(np.abs(cross_correlation[tau_range])) +
                    eps) / np.sqrt((P_l + eps) * (P_r + eps))
        ILD[b] = 10 * np.log10((P_l + eps) / (P_r + eps))

        # Compute ITD using quadratic interpolation
        spl = CubicSpline(tau_range, cross_correlation[tau_range])
        factor = 4
        tau_range_new = np.arange(int(-fs * tau_limit), int(fs * tau_limit), step=1/factor)
        crosscorr_new = spl(tau_range_new)
        ITD[b] = (np.argmax(crosscorr_new) / (fs*factor)) - tau_limit

        P_L[b] = P_l
        P_R[b] = P_r

    return IC, ITD, ILD, P_L, P_R

