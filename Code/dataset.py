import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm import tqdm
from scipy.signal import resample
from scipy.signal import hilbert
from scipy.signal import welch, correlate
from scipy.stats import ttest_rel

from utils import *
from constants import *

class Participant:
    def __init__(self, name, data_path=DATA_PATH, alpha=ALPHA):
        self.name = name
        data = load_pickle(data_path)
        self.sessions = [
            Session(sess,
                    data[self.name][sess]['fs'],
                    data[self.name][sess]['neural_data'],
                    data[self.name][sess]['trials_info'])
                for sess in data[self.name].keys()
            ]
        
        # CHANNELS 
        self.channels_locations = data[self.name]['sess1']['channel_locations'][0]
        self.channels_labels = data[self.name]['sess1']['channel_labels']
        self.nb_channels = len(self.channels_locations)
        self.channels = [
            Channel(self.channels_labels[i], i, self.channels_locations[i], self.name, self.sessions)
            for i in range(self.nb_channels)
        ]
        self.relevant_channels = [channel for channel in self.channels if channel.p_value_Ex < alpha and channel.p_value_Obs < alpha]

    def get_features_per_session_ExObs(self, session, freq_band=FREQ_BANDS, features=FEATURES, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
        trial_dict = {}
        trial_dict['label'] = []
        for trial in tqdm(session.trials):
            trial_dict['label'].append(trial.action_type)
            trial_dict = self.get_features(trial_dict, trial, freq_band, features, window_size, step_size)
                                       
        return pd.DataFrame(trial_dict)

    def get_features_all_sessions_ExObs(self, freq_band=FREQ_BANDS, features=FEATURES, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
        return pd.concat([
            self.get_features_per_session_ExObs(session, freq_band, features, window_size, step_size)
            for session in self.sessions
        ])
        
    def get_features_per_session_mvt(self, session, data, freq_band=FREQ_BANDS, features=FEATURES, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
        if not (data=='E' or data=='O'): raise ValueError('Invalid data')
        trial_dict = {}
        trial_dict['label'] = []
        for trial in tqdm(session.trials):
            if trial.action_type != data: continue
            trial_dict['label'].append(trial.object_size)
            trial_dict = self.get_features(trial_dict, trial, freq_band, features, window_size, step_size) 
                                       
        return pd.DataFrame(trial_dict)
    
    def get_features_all_sessions_mvt(self, data, freq_band=FREQ_BANDS, features=FEATURES, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
        if not (data!='E' or data!='O'): raise ValueError('Invalid data')
        return pd.concat([
            self.get_features_per_session_mvt(session, data, freq_band, features, window_size, step_size)
            for session in self.sessions
        ])
    
    def get_features(self, trial_dict, trial, freq_band, features, window_size, step_size):
        for channel in tqdm(self.relevant_channels):
            for band_name, (low, high) in freq_band.items():
                signal = trial.get_preprocessed_signal(channel, low, high)
                for feature in features:
                    n = len(signal)
                    for i, start in enumerate(range(0, n - window_size + 1, step_size)):
                        value = feature(signal[start:start + window_size])
                        label = f'CH{channel.idx}_{band_name}_{feature.__name__}_window_{i}'
                        if label in trial_dict.keys():
                            trial_dict[label].append(value)
                        else:
                            trial_dict[label] = [value] 
        return trial_dict
    
    def plot_channel_responsiveness(self, dB=True):
        
        num_cols = 4
        fig_width = 20
        num_rows = int(np.ceil(self.nb_channels / num_cols))
        fig_height = fig_width * num_rows / num_cols
        _, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), sharey=False)
        axs = axs.flatten()
        
        for channel_id in tqdm(range(self.nb_channels)):
            channel = self.channels[channel_id]
            if dB:
                axs[channel_id].plot(channel.pds_baseline_Ex[0], 10*np.log10(channel.pds_baseline_Ex[1]), label='baseline Ex', alpha=0.5)
                axs[channel_id].plot(channel.pds_aroundObjGrasped_Ex[0], 10*np.log10(channel.pds_aroundObjGrasped_Ex[1]), label='aroundObjGrasped Ex', alpha=0.5)
                axs[channel_id].plot(channel.pds_aroundObjGrasped_Ex[0], np.log10(channel.pds_baseline_Ex[1])/np.log10(channel.pds_aroundObjGrasped_Ex[1]), label='absolute difference Ex')
                axs[channel_id].plot(channel.pds_baseline_Obs[0], 10*np.log10(channel.pds_baseline_Obs[1]), label='baseline Obs', alpha=0.5)
                axs[channel_id].plot(channel.pds_aroundObjGrasped_Obs[0], 10*np.log10(channel.pds_aroundObjGrasped_Obs[1]), label='aroundObjGrasped Obs', alpha=0.5)
                axs[channel_id].plot(channel.pds_aroundObjGrasped_Obs[0], np.log10(channel.pds_baseline_Obs[1])/np.log10(channel.pds_aroundObjGrasped_Obs[1]), label='absolute difference Obs')         
            else:
                axs[channel_id].plot(channel.pds_baseline_Ex[0], channel.pds_baseline_Ex[1], label='baseline Ex', alpha=0.5)
                axs[channel_id].plot(channel.pds_aroundObjGrasped_Ex[0], channel.pds_aroundObjGrasped_Ex[1], label='aroundObjGrasped Ex', alpha=0.5)
                axs[channel_id].plot(channel.pds_aroundObjGrasped_Ex[0], abs(channel.pds_baseline_Ex[1] - channel.pds_aroundObjGrasped_Ex[1]), label='absolute difference Ex')
                axs[channel_id].plot(channel.pds_baseline_Obs[0], channel.pds_baseline_Obs[1], label='baseline Obs', alpha=0.5)
                axs[channel_id].plot(channel.pds_aroundObjGrasped_Obs[0], channel.pds_aroundObjGrasped_Obs[1], label='aroundObjGrasped Obs', alpha=0.5)
                axs[channel_id].plot(channel.pds_aroundObjGrasped_Obs[0], abs(channel.pds_baseline_Obs[1] - channel.pds_aroundObjGrasped_Obs[1]), label='absolute difference Obs')
            
            if channel in self.relevant_channels:
                axs[channel_id].set_title(f'Power spectral density for channel {channel_id} - RESPONSIVE')
            else:
                axs[channel_id].set_title(f'Power spectral density for channel {channel_id}')
            
            axs[channel_id].set_xlabel('frequency [Hz]')
            axs[channel_id].set_ylabel('PSD [V**2/Hz]')
            axs[channel_id].set_xlim(0, 150)
            axs[channel_id].legend()
            axs[channel_id].grid()
            
        plt.tight_layout()
        plt.show()
        
    def plot_preprocessed_signal(self, trial=None):
        
        num_cols = len(FREQ_BANDS)
        fig_width = 20
        num_rows = int(len(self.relevant_channels))
        fig_height = fig_width * num_rows / num_cols
        _, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), sharey=False)
                
        if not trial:
            trial = self.sessions[0].trials[0]
        
        for i, channel in enumerate(self.relevant_channels):
            for j, band in enumerate(FREQ_BANDS.keys()):
                signal = trial.get_preprocessed_signal(channel, FREQ_BANDS[band][0], FREQ_BANDS[band][1])
                axs[i, j].plot(signal)
                axs[i, j].set_title(f'Channel {channel.idx} - {band}')
                axs[i, j].grid()
                
        plt.tight_layout()
        plt.show()
        
class Channel:
    def __init__(self, name, idx, location, participant, sessions):
        self.name = name
        self.idx = idx
        self.location = location
        self.participant = participant
        
        pds_baseline_Ex = []
        pds_aroundObjGrasped_Ex = []
        
        pds_baseline_Obs = []
        pds_aroundObjGrasped_Obs = []
        
        for session in sessions:
            for trial in session.trials:
                if trial.action_type == 'E':
                    pds_baseline_Ex.append(trial.get_pds_baseline(idx))
                    pds_aroundObjGrasped_Ex.append(trial.get_pds_aroundObjGrasped(idx))
                if trial.action_type == 'O':
                    pds_baseline_Obs.append(trial.get_pds_baseline(idx))
                    pds_aroundObjGrasped_Obs.append(trial.get_pds_aroundObjGrasped(idx))
                
        self.pds_baseline_Ex = np.mean(np.array(pds_baseline_Ex), axis=0)
        self.pds_aroundObjGrasped_Ex = np.mean(np.array(pds_aroundObjGrasped_Ex), axis=0)
        self.pds_baseline_Obs = np.mean(np.array(pds_baseline_Obs), axis=0)
        self.pds_aroundObjGrasped_Obs = np.mean(np.array(pds_aroundObjGrasped_Obs), axis=0)
        
        self.absolute_psd_difference_Ex = np.abs(self.pds_aroundObjGrasped_Ex - self.pds_baseline_Ex)
        self.corr_Ex = correlate(self.pds_baseline_Ex, self.pds_aroundObjGrasped_Ex)
        self.t_stat_Ex, self.p_value_Ex = ttest_rel(self.pds_baseline_Ex[1], self.pds_aroundObjGrasped_Ex[1])
        
        self.absolute_psd_difference_Obs = np.abs(self.pds_aroundObjGrasped_Obs - self.pds_baseline_Obs)
        self.corr_Obs = correlate(self.pds_baseline_Obs, self.pds_aroundObjGrasped_Obs)
        self.t_stat_Obs, self.p_value_Obs = ttest_rel(self.pds_baseline_Obs[1], self.pds_aroundObjGrasped_Obs[1])

    
class Session: 
    def __init__(self, name, fs, neural_data, trials_info):
        self.name = name
        self.fs = fs
        self.neural_data = neural_data
        
        # TRIALS
        self.no_error = np.array(trials_info['ErrorCode']) == 0
        self.nb_trials = len(trials_info['TS_TrialStart'])
        self.trials = [
            Trials(i, self.fs,  self.neural_data, trials_info)
            for i in range(self.nb_trials)
            if self.no_error[i]
        ]
        
class Trials: 
    def __init__(self, trial_idx, fs, neural_data, trials_info):
        trial_baseline_start = trials_info['TS_TrialStart'][trial_idx]
        trial_baseline_stop = trials_info['TS_CueOn'][trial_idx]
        self.trial_start = trials_info['TS_TrialStart'][trial_idx]
        trial_stop = trials_info['TS_HandBack'][trial_idx]
        
        self.trial_idx = trial_idx
        self.trials_info = trials_info
        self.fs = fs
        self.action_type = trials_info['ActionType'][trial_idx]
        self.object_size = trials_info['ObjectSize'][trial_idx]
        
        signal = neural_data[:, int(self.trial_start * fs):int(trial_stop * fs)+1]
        signal = bandpass_filter(signal, 0.5, 150, fs)
        signal = noise_filter(signal, fs)
        self.signal = signal
        
        baseline = neural_data[:, int(trial_baseline_start * fs):int(trial_baseline_stop * fs)]
        baseline = bandpass_filter(baseline, 0.5, 150, fs)
        baseline = noise_filter(baseline, fs)
        self.baseline_signal = baseline
        
    def get_mean_baseline(self):
        return np.mean(self.baseline_signal, axis=1)
    
    def get_subsampled_baseline(self, subsampling_frequency=SUBSAMPLING_FREQUENCY):
        return subsample(self.baseline_signal, self.fs, subsampling_frequency)
    
    def get_signal(
            self, trigger_start='TS_HandOut', trigger_stop='TS_HandBack',
            subsampling_frequency=SUBSAMPLING_FREQUENCY, baseline_correction=True, nb_samples=NB_SAMPLES):
        
        if trigger_start not in self.trials_info.keys() or trigger_stop not in self.trials_info.keys():
            raise ValueError('Invalid trigger')
        
        start_idx = int(self.trials_info[trigger_start][self.trial_idx] * self.fs) - int(self.trial_start * self.fs)
        stop_idx = int(self.trials_info[trigger_stop][self.trial_idx] * self.fs) - int(self.trial_start * self.fs)
        signal = self.signal[:, start_idx:stop_idx]

        if subsampling_frequency:
            signal = subsample(signal, self.fs, subsampling_frequency)
        if baseline_correction:
            signal = signal - np.outer(self.get_mean_baseline(), np.ones(np.shape(signal)[1]))
        if nb_samples:
            signal = resample(signal, NB_SAMPLES, axis=1)

        return signal
    
    def get_preprocessed_signal(self, channel, low, high):
        signal = self.get_signal()[channel.idx]
        signal = bandpass_filter(signal, low, high, self.fs)
        signal = hilbert(signal)
        signal = np.abs(signal)
        return signal
    
    def get_pds_baseline(self, channel_idx):
        return welch(self.baseline_signal[channel_idx], fs=self.fs, nperseg=self.fs/2)
    
    def get_pds_aroundObjGrasped(self, channel_idx):
        objectGrasp = self.trials_info['TS_ObjectGrasp'][self.trial_idx]
        start_idx = int((objectGrasp - 0.5) * self.fs) - int(self.trial_start * self.fs)
        stop_idx = int((objectGrasp + 0.5) * self.fs) - int(self.trial_start * self.fs)
        return welch(self.signal[channel_idx, start_idx:stop_idx], fs=self.fs, nperseg=self.fs/2)
    
def subsample(signal, fs, subsampling_frequency):
    step = fs // subsampling_frequency
    return signal[:, ::step]

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the input signal.

    Args:
        signal (ndarray): (nb_channels, nb_timepoints) NumPy array, the signal to filter.
        lowcut (float): lower frequency of the band (Hz).
        highcut (float): higher frequency of the band (Hz).
        fs (float): sampling frequency of the signal (Hz). 
        order (int, optional): the order of the filter. Defaults to 4.

    Returns:
        filtered_data (ndarray): (nb_channels, nb_timepoints) NumPy array, the filtered signal.
    """
    if lowcut >= highcut:
        raise ValueError("Lowcut frequency must be less than highcut frequency.")
        
    if fs <= 2 * highcut:
        raise ValueError("Sampling frequency must respect the Nyquist criteria.")
        
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, signal)

def noise_filter(signal, fs, nb_harmonics=2):
    # removing 50Hz noise and its harmonics
    powergrid_noise_frequencies_Hz = [harmonic_idx*50 for harmonic_idx in range(1,nb_harmonics+1)] 

    for noise_frequency in powergrid_noise_frequencies_Hz:
        sos = butter(N=4, Wn=(noise_frequency - 2, noise_frequency + 2), fs=fs, btype="bandstop", output="sos")
        signal = sosfiltfilt(sos, signal)
        
    return signal

# def get_features_across_participants_ExObs(participants=PARTICIPANTS):
#     for participant in participants:
#         participant = Participant(participant)
#         df = participant.get_features_all_sessions_ExObs()
        

if __name__ == '__main__':
    p = Participant('s6')
    # df = p.get_features_all_sessions_ExObs()
    df = p.get_features_all_sessions_mvt(data='E')
    


