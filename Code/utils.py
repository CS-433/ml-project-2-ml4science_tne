import pickle

from scipy.signal import butter, sosfiltfilt

from constants import FS

def load_pickle(data_path):
    filename = data_path +  '.pickle'
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def get_max_nb_channels(df):
    """
    Get the highest number of channels throughout participants and sessions

    Args:
        data (DataFrame): dataset

    Returns:
        int: highest number of channels throughout participants and sessions
    """
    
    return max(df['neural_data'].apply(lambda x: len(x)))
    
def bandpass_filter(data, lowcut, highcut, fs=FS, order=4):
    """
    Apply a bandpass filter to the input data.

    Args:
        data (ndarray): 1D NumPy array, the signal to filter.
        lowcut (float): lower frequency of the band (Hz).
        highcut (float): higher frequency of the band (Hz).
        fs (float): sampling frequency of the signal (Hz). Defaults to FS.
        order (int, optional): the order of the filter. Defaults to 4.

    Returns:
        filtered_data (ndarray): 1D NumPy array, the filtered signal.
    """
    if lowcut >= highcut:
        raise ValueError("Lowcut frequency must be less than highcut frequency.")
        
    if fs <= 2 * highcut:
        raise ValueError("Sampling frequency must respect the Nyquist criteria.")
        
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data)
    
def bandstop_filter(data, lowcut, highcut, fs=FS, order=4):
    """
    Apply a bandstop filter to the input data.

    Args:
        data (ndarray): 1D NumPy array, the signal to filter.
        lowcut (float): lower frequency of the band (Hz).
        highcut (float): higher frequency of the band (Hz).
        fs (float): sampling frequency of the signal (Hz). Defaults to FS.
        order (int, optional): the order of the filter. Defaults to 4.
        
    Returns:
        filtered_data (ndarray): 1D NumPy array, the filtered signal.
    """
    if lowcut >= highcut:
        raise ValueError("Lowcut frequency must be less than highcut frequency.")
        
    if fs <= 2 * highcut:
        raise ValueError("Sampling frequency must respect the Nyquist criteria.")
    
    sos = butter(order, [lowcut, highcut], btype='bandstop', fs=fs, output='sos')
    return sosfiltfilt(sos, data)
    
def compute_acc(preds, y):
    pass

def compute_F1(preds, y):
    pass