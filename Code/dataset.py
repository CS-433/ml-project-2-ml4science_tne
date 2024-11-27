import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d

from scipy.signal import welch, correlate, coherence
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

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

def get_trials(session_data, fps):
    '''
    Get the trials from the session data.
    
    Parameters:
    - session_data: dict, the data of a session.
    - fps: int, the sampling rate of the data.
    
    Returns:
    - trials: list of 2D NumPy arrays, each element in the list is an array with the timeseries of the trial in each channel.
    '''
    trial_starts = session_data['trials_info']['TS_TrialStart']
    trial_end = session_data['trials_info']['TS_HandBack']
    trials = []
    for i in range(len(trial_starts)):
        trials.append(session_data['neural_data'][:,int(trial_starts[i]) * fps:int(trial_end[i]) * fps])
    return trials

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

def get_baselines_and_activities(session_data, channel_id, fps=2048):

    trial_starts = session_data['trials_info']['TS_TrialStart']
    trial_action = session_data['trials_info']['TS_ObjectGrasp']
    no_errors = np.array(session_data['trials_info']['ErrorCode']) == 0

    channel_ts = session_data['neural_data'][channel_id]
    channel_baselines = []
    channel_actions = []
    for i in range(len(trial_starts)):
        if no_errors[i]:
            channel_baselines.append(channel_ts[int((trial_starts[i]) * fps):int((trial_starts[i] + 1) * fps)])
            channel_actions.append(channel_ts[int((trial_action[i] - 0.5) * fps):int((trial_action[i] + 0.5) * fps)])
    
    return channel_baselines, channel_actions

def get_mean_baseline_and_activity (session_data, channel_id, fps=2048):

    channel_baselines, channel_actions = get_baselines_and_activities(session_data, channel_id, fps)
        
    mean_baseline = np.array(channel_baselines).mean(axis=0)
    mean_action = np.array(channel_actions).mean(axis=0)
    
    return mean_baseline, mean_action


def analyze_differences_between_baseline_and_activity (session, num_channels, plot, fps=2048, num_cols = 4, fig_width = 20):
    '''
    Analyze the differences between the baseline and activity for each channel in the session data.
    This function calculates the power spectral density, correlation, coherence, t-test between the baseline signal and activity signal for each channel.
    It can also plot the power spectral density of the baseline and activity as well as their difference for each channel.
    
    Parameters:
    - session: dict, the data of a session.
    - num_channels: int, the number of channels to analyse.
    - plot: bool, whether to plot the power spectral density or not.
    - fps: int, the sampling rate of the data. (default 2048)
    - num_cols: int, the number of columns plot. (default 4)
    - fig_width: int, the width of the figure. (default 20)
    
    Returns:
    - list_baseline_psds: list of tuples, the power spectral density of the baseline for each channel.
                          The first element of the tuple is the frequency and the second is the power spectral density at that given frequency.
    - list_activity_psds: list of tuples, the power spectral density of the activity for each channel.
                          The first element of the tuple is the frequency and the second is the power spectral density at that given frequency.
    - absolute_psd_difference: list of 1D NumPy arrays, the absolute difference between the power spectral density of the baseline and activity for each channel.
                               Remark: Uses the same frequencies as the power spectral density of the baseline and the power spectral density of the activity.
    - corr: list of 1D numpy arrays, the correlation funcion between the baseline and activity for each channel. 
            Corresponds to the correlation of the both signals in the  time domain.
    - cohe: list of tuples of numpy arrays, the coherence function between the baseline and activity for each channel. 
            The first element of the tuple is the frequency and the second is the coherence at that given frequency.
    - ttests: list of tuples, the t-test between the power spectral density of the baseline and activity for each channel.
    '''
    
    num_rows = np.ceil(num_channels / num_cols).astype(int)
    fig_height = fig_width * num_rows / num_cols

    if plot : 
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    plot_ids = np.ones((num_rows, num_cols), dtype=object)

    list_baseline_psds = []
    list_activity_psds = []    
    absolute_psd_difference = []
    ttests = []
    corr = []
    cohe = []
    

    for i,row in enumerate(plot_ids):
        for j, _ in enumerate(row):
            if i*num_cols+j >= num_channels:
                break
            mean_baseline, mean_activity = get_mean_baseline_and_activity(session,i*num_cols+j)
            baseline_psd = welch(mean_baseline, fs=fps)
            list_baseline_psds.append(baseline_psd)
            list_activity_psds.append(welch(mean_activity, fs=fps))
            activity_psd = welch(mean_activity, fs=fps)
            absolute_psd_difference.append(np.abs(activity_psd[1] - baseline_psd[1]))
            corr.append(correlate(mean_baseline, mean_activity))
            cohe.append(coherence(mean_baseline, mean_activity, fs=fps))
            ttests.append(ttest_rel(baseline_psd[1], activity_psd[1]))
            if plot:
                axs[i][j].plot(baseline_psd[0],baseline_psd[1], label='baseline', alpha=0.5)
                axs[i][j].plot(activity_psd[0],activity_psd[1], label='activity', alpha=0.5)
                axs[i][j].plot(baseline_psd[0],absolute_psd_difference[-1], label='absolute difference', alpha=0.5)
                axs[i][j].set_title(f'Power spectral density for channel {i*num_cols+j}')
                axs[i][j].set_xlabel('frequency [Hz]')
                axs[i][j].set_ylabel('PSD [V**2/Hz]')
                axs[i][j].set_xlim(0, 150)
                axs[i][j].legend()
                axs[i][j].grid()
    if plot:
        plt.tight_layout()
        plt.show()
    
    return list_baseline_psds, list_activity_psds, absolute_psd_difference, corr, cohe, ttests