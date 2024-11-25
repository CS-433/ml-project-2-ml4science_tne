import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d

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
    # TODO find a better way to remove all electrical noise in one sweep
    elec_noise = [(k*50-2, k*50+2) for k in range(1, 3)]
    # Apply bandstop filter to remove electrical noise
    # For each participant
    for participant in data.keys():
        # For each session
        for session in data[participant].keys():
            # For each channel in list of channels
            for channel in range(len(data[participant][session]['neural_data'])):
                # For each noise freq.
                for noise_freq in elec_noise:
                    data[participant][session]['neural_data'][channel] = \
                        bandstop_filter(data[participant][session]['neural_data'][channel], noise_freq[0], noise_freq[1])

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
    # baseline_data = separate_segments(df, 'TS_TrialStart', 'TS_CueON')
    # # Is experiment going from TrialStart, CueOn or GoSignal
    # experiment_data = separate_segments(df, 'TS_CueOn', 'TS_HandBack')
    
    # if method == 'mean':
    #     baseline = np.mean(baseline_data)
    # elif method == 'single':
    #     baseline = np.median(baseline_data)
    # else:
    #     raise ValueError('Invalid method')
    
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
    
    dfc = df.copy(deep=True)
    dfc['sub_neural_data'] = df['neural_data'].apply(lambda x: x[::subsampling_frequency])
    dfc = dfc.drop(['neural_data'], axis=1)
    return dfc

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
    max_nb_channels = get_max_nb_channels(dfc)
    
    def sep_signal_in_freqs(row):
        signal = row['neural_data']
    
        new_cols = []
        for channel_idx in range(max_nb_channels):
            if channel_idx < len(signal):
                # Remove frequencies that are too low (noise) or too high (not eeg-related)
                channel = bandpass_filter(signal[channel_idx], 0.5, 150)
                for band_name, (low, high) in freq_bands.items():
                    # Bandpass filter the channel
                    filtered_channel = bandpass_filter(channel, low, high)
                    # Take the envelope of the frequency band with a Hilbert transform
                    enveloped_channel = hilbert(rectified_channel)
                    enveloped_channel = np.abs(enveloped_channel)
                    new_cols.append(enveloped_channel)
            else:
                for _ in range(len(freq_bands)):
                    new_cols.append(None)
                
        return tuple(new_cols)
    
    new_cols = ([f'channel{channel}_{band_name}' for channel in range(max_nb_channels) for band_name in freq_bands.keys()])
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

def get_trials(session_data, channel, fps):
    '''
    Get the trials from the session data for a given channel.
    
    Parameters:
    - session_data: dict, the data of a session.
    - channel: int, the index of the channel.
    - fps: int, the sampling rate of the data.
    
    Returns:
    - trials: list of 1D NumPy arrays, the trials.
    '''
    
    trial_starts = session_data['trials_info']['TS_TrialStart']
    trial_end = session_data['trials_info']['TS_HandBack']
    trials = {f"trial_{i}":session_data['neural_data'][channel][int(trial_starts[i]) * fps:int(trial_end[i]) * fps] for i in range(len(trial_starts))}
    return trials

def separate_trials (session_data, fps=2048) :
    '''
    Separate trials from the session data for all channels.
    
    Parameters:
    - session_data: dict, the data of a session.
    - fps: int, the sampling rate of the data.
    
    Returns:
    - session_data: dict, the input dictionnary with an new trial key that stores the timepoints for each trial.
    '''
    trial_dict = {f'channel_{channel}': get_trials(session_data, channel, fps = fps) for channel in range(len(session_data['channel_labels']))}
    session_data['trials'] = trial_dict
    return session_data

def remove_trials_with_error (session_data):
    '''
    Remove the trials with an error from the session data.
    
    Parameters:
    - session_data: dict, the data of a session.
    
    Returns:
    - session_data: dict, the input dictionnary with the erroneous trials removed
    '''
    
    erroneous_trials = [i for i in range(len(session_data['trials_info']['ErrorType'])) if session_data['trials_info']['ErrorType'][i] != 'NoError']

    for trial_id in erroneous_trials:
        for channel_idx in range(len(session_data['channel_labels'])):
            session_data['trials'][f'channel_{channel_idx}'].pop(f'trial_{trial_id}')
                
    return session_data

def substract_mean_baseline (session_data, fps, baseline_duration = 1.0, how='each_trial'):
    '''
    Normalize the entire signal with the average of the baseline periods over all trials.
    
    Parameters:
    - session_data: dict, the session data.
    - fps: int, the sampling rate of the data.
    - baseline_duration: int, the duration of the baseline period in seconds.
    - how: string, the normalization method to use. Either 'each_trial' or 'all_trials'.
    
    Returns:
    - session_data: dict, the input dictionary with the trials normalized.
    '''
    if how == 'each_trial':
        for channel in session_data['trials'].keys():
            for trial in session_data['trials'][channel].keys():
                mean = session_data['trials'][channel][trial][0:int(fps*baseline_duration)].mean()
                session_data['trials'][channel][trial] = session_data['trials'][channel][trial] - mean
                
    elif how == 'all_trials':
        means = []
        for channel in session_data['trials'].keys():
            for trial in session_data['trials'][channel].keys():
                means.append(session_data['trials'][channel][trial][0:int(fps*baseline_duration)].mean())
        total_mean = np.mean(means)
        for channel in session_data['trials'].keys():
            for trial in session_data['trials'][channel].keys():
                session_data['trials'][channel][trial] = session_data['trials'][channel][trial] - total_mean
    
    return session_data    