import numpy as np

# Define modified logistic function mapping 
def scaled_sigmoid(x, x0=0.5, k=10, y_min=0.00005, y_max=0.001):
    psi_min = 1 / (1 + np.exp(-k * (0 - x0)))  # Sigmoid at x=0
    psi_max = 1 / (1 + np.exp(-k * (1 - x0)))  # Sigmoid at x=1
    
    psi_x = 1 / (1 + np.exp(-k * (x - x0)))  # Original sigmoid function
    y_x = y_min + (psi_x - psi_min) * (y_max - y_min) / (psi_max - psi_min)  # Scale to [y_min, y_max]
    return y_x