import os
import numpy as np

# Data path
DATA_PATH = os.path.join(os.getcwd(), 'Data', 'Dataset_4subjects_Exe_Obs')
DATA_PATH_NOTEBOOK = os.path.join(os.getcwd(), '..', 'Data', 'Dataset_4subjects_Exe_Obs')
# Sampling frequency in Hz (same for all)
FS = 2048
# Subsampling frequency in Hz
SUBSAMPLING_FREQUENCY = 500
# Number of samples per trial for duration normalization
NB_SAMPLES = 1500
# Alpha for the relevant channels
ALPHA = 0.05
# Frequency bands for separating data by frequency
FREQ_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
    "HighGamma": (100, 150)
}
# Features 
FEATURES = [np.mean, np.std]
# Moving average
WINDOW_SIZE = 1000
STEP_SIZE = 100
# Participants
PARTICIPANTS = ['s6', 's7', 's11', 's12']
# Reproducibility
RANDOM_STATE = 12