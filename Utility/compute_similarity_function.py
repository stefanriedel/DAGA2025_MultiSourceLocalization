import numpy as np

def compute_similarity_function(StimulusGradient_L, StimulusGradient_R, Mem_Grad_L, Mem_Grad_R, freq_centers,  low_band, high_band):
    num_bands =  StimulusGradient_L.shape[0]

    MemoryGradient_L = Mem_Grad_L
    MemoryGradient_R = Mem_Grad_R
    num_directions = Mem_Grad_L.shape[1]

    Sim_Func_L = np.zeros(num_directions)
    Sim_Func_R = np.zeros(num_directions)

    # Start computation for the provided binaural stimulus
    b1 = low_band
    b2 = high_band

    DeltaSpecGrad_L = np.zeros(num_directions)
    DeltaSpecGrad_R = np.zeros(num_directions)

    # Classical variant according to Baumgartner model
    for d in range(num_directions):
        DeltaSpecGrad_L[d] = np.mean(np.abs(MemoryGradient_L[b1:,d] - StimulusGradient_L[b1:])) 
        DeltaSpecGrad_R[d] = np.mean(np.abs(MemoryGradient_R[b1:,d] - StimulusGradient_R[b1:])) 
    Gamma = 6
    S = 0.5
    Sim_Func_L = 1 - (1 + np.exp(-Gamma * (DeltaSpecGrad_L - S)))**-1
    Sim_Func_R = 1 - (1 + np.exp(-Gamma * (DeltaSpecGrad_R - S)))**-1

    return Sim_Func_L, Sim_Func_R