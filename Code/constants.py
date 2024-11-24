# List of bad channels that need to be removed from the dataset
BAD_CHANNELS = []
# Sampling frequency in Hz (same for all)
FS = 2048
# Subsampling frequency in Hz
SUBSAMPLING_FREQUENCY = 500
# Frequency bands for separating data by frequency
FREQ_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
    "HighGamma": (100, 150)
}