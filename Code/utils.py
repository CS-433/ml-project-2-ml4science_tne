import pickle

from scipy.signal import butter, sosfilt

def load_pickle(data_path):
    filename = data_path +  '.pickle'
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def get_highest_number_of_channels(df):
    """
    Get the highest number of channels throughout participants and sessions

    Args:
        data (DataFrame): dataset

    Returns:
        int: highest number of channels throughout participants and sessions
    """
    
    return max(df['neural_data'].apply(lambda x: len(x)))
    
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the input data.

    Parameters:
    - data: 1D NumPy array, the signal to filter.
    - lowcut: float, lower frequency of the band (Hz).
    - highcut: float, upper frequency of the band (Hz).
    - fs: float, sampling frequency of the signal (Hz).
    - order: int, the order of the filter.

    Returns:
    - filtered_data: 1D NumPy array, the filtered signal.
    """
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfilt(sos, data)