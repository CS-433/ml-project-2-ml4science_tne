import numpy as np
import pandas as pd

from utils import *
from constants import *

def get_dataset(data_path='../Data/Dataset_4subjects_Exe_Obs', type='without_normalization', norm_before_sep=False, norm_after_sep=False):
    """
    Load and preprocess the dataset

    Args:
        data_path (str, optional): _description_. Defaults to '../Data/Dataset_4subjects_Exe_Obs'.
        type (str, optional): _description_. Defaults to 'without_normalization'.

    Returns:
        _type_: _description_
    """
    
    data = load_pickle(data_path)
    data = remove_bad_channels(data)
    data = remove_electrical_noise(data)
    
    df_trials = separate_trials(data)
    df_trials = baseline_correction(df_trials)
    
    if norm_before_sep:
        df_trials = normalize(df_trials)
        
    df_trials = subsample(df_trials)
    
    if norm_after_sep:
        df_trials = normalize(df_trials)
        
    df_trials = separate_frequency_bands(df_trials)
    df_trials = standardize_duration(df_trials)
    return df_trials
    
def remove_bad_channels(data, bad_channels=BAD_CHANNELS):
    """
    Remove bad channels from the dataset

    Args:
        df (dictionary): dataset
        bad_channels (list, optional): list of bad channels. Defaults to BAD_CHANNELS.

    Returns:
        dictionary: dataset without bad channels 
    """
    
    return data
    
    
def remove_electrical_noise(data):
    """
    Remove electrical noise from the dataset

    Args:
        data (dictionary): dataset

    Returns:
        dictionary: dataset with applied notch filter to remove electrical noise
    """
    
    return data
    
def separate_trials(data):
    """
    Separate the dataset into trials & remove trials with errors

    Args:
        data (dictionary): dataset

    Returns:
        dictionary: dataframe with separated trials (participants, session, obs/ex, neural_data)
    """
    
    
    df = pd.DataFrame(data)
    
    return df
    
def baseline_correction(df, method='single'):
    """
    Apply baseline correction to the dataset

    Args:
        data (DataFrame): dataset
        method ('single'|'mean'): method to apply the baseline correction. Defaults to 'single'.

    Returns:
        DataFrame: dataset with applied baseline correction
    """
    baseline_data = separate_segments(df, 'TS_TrialStart', 'TS_CueON')
    # Is experiment going from TrialStart, CueOn or GoSignal
    experiment_data = separate_segments(df, 'TS_CueOn', 'TS_HandBack')
    
    if method == 'mean':
        baseline = np.mean(baseline_data)
    elif method == 'single':
        baseline = np.median(baseline_data)
    else:
        raise ValueError('Invalid method')
    
    return df
    
def separate_segments(df, trigger1, trigger2):
    """
    Separate the dataset into segments

    Args:
        data (DataFrame): dataset
        trigger1 (str): first trigger
        trigger2 (str): second trigger

    Returns:
        DataFrame: dataset with separated segments
    """
    
    return df

def normalize(df):
    """
    Normalize the dataset

    Args:
        data (DataFrame): dataset

    Returns:
        DataFrame: dataset with normalized data
    """
    
    return df

def subsample(df, subsampling_frequency=SUBSAMPLING_FREQUENCY):
    """
    Subsample the dataset

    Args:
        data (DataFrame): dataset
        subsampling_frequency (int, optional): subsampling frequency. Defaults to SUBSAMPLING_FREQUENCY.

    Returns:
        DataFrame: dataset with subsampled data
    """
    
    return df

def separate_frequency_bands(df, freq_bands=FREQ_BANDS):
    """
    Separate the dataset into frequency bands

    Args:
        data (DataFrame): dataset
        freq_bands (dict, optional): frequency bands. Defaults to FREQ_BANDS.

    Returns:
        DataFrame: dataset with separated frequency bands & channels (participants, session, obs/ex, channel1_alpha, ...)
    """
    dfc = df.copy()
    max_nb_channels = get_highest_number_of_channels(dfc)
    new_cols = ([f'channel{channel}_{band_name}' for channel in range(max_nb_channels) for band_name in freq_bands.keys()])
    
    def sep_signal_in_freqs(row):
        signal = row['neural_data']
    
        new_cols = []
        for channel in range(max_nb_channels):
            if channel < len(signal):
                for band_name, (low, high) in freq_bands.items():
                    filtered_channel = bandpass_filter(signal[channel], low, high, 500)
                    new_cols.append(filtered_channel)
            else:
                for _ in range(len(freq_bands)):
                    new_cols.append(None)
                
        return tuple(new_cols)
    
    dfc[new_cols] = dfc.apply(sep_signal_in_freqs, axis=1, result_type='expand')
    dfc = dfc.drop(['neural_data'], axis=1)
    return dfc


def standardize_duration(df):
    """
    Standardize the duration of the dataset

    Args:
        data (DataFrame): dataset

    Returns:
        DataFrame: dataset with standardized duration
    """
    
    return df
