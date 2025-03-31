import numpy as np

def compute_block_gradients(P_X_L, P_X_R, pos_grad=True, freq_dependent=True, delta_dB=20, energy_weighting=True):
    delta_dB = 30
    num_bands = P_X_L.size

    SpecGrad_L = np.zeros(num_bands)
    SpecGrad_R = np.zeros(num_bands)

    Mag_L = np.zeros(num_bands)
    Mag_R = np.zeros(num_bands)

    for b in range(1,num_bands):
        P_L = P_X_L[b]
        P_R = P_X_R[b]

        Mag_L[b] = 10*np.log10(P_L)
        Mag_R[b] = 10*np.log10(P_R)

        if pos_grad:
            SpecGrad_L[b] = np.maximum(Mag_L[b] - Mag_L[b-1], 0) 
            SpecGrad_R[b] = np.maximum(Mag_R[b] - Mag_R[b-1], 0)
        else:
            SpecGrad_L[b] = Mag_L[b] - Mag_L[b-1]
            SpecGrad_R[b] = Mag_R[b] - Mag_R[b-1]

    if energy_weighting:
        for b in range(1,num_bands):
            SpecGrad_L[b] *= np.minimum(10**(-(np.max(Mag_L)-delta_dB-Mag_L[b])/10), 1)
            SpecGrad_R[b] *= np.minimum(10**(-(np.max(Mag_R)-delta_dB-Mag_R[b])/10), 1)

    return SpecGrad_L, SpecGrad_R