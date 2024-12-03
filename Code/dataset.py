import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.signal import resample
from scipy.signal import hilbert

from scipy.signal import welch, correlate, coherence
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from matplotlib import ticker

from utils import *
from constants import *

class participant:
    def __init__(self, name, data_path=DATA_PATH, alpha=ALPHA):
        self.name = name
        data = load_pickle(data_path)
        self.sessions = [session(sess, data[self.name][sess]['fs'], data[self.name][sess]['neural_data'], data[self.name][sess]['trials_info']) for sess in data[self.name].keys()]
        
        # CHANNELS 
        self.channels_locations = data[self.name]['sess1']['channel_locations']
        self.channels_labels = data[self.name]['sess1']['channel_labels']
        self.nb_channels = len(self.channels_locations)
        self.channels = [channel(self.channels_labels[i], i, self.channels_locations[i], self.name, self.sessions) for i in range(self.nb_channels)]
        self.relevant_channel = [channel.idx for channel in self.channels if channel.p_value < alpha]
        
    def get_features_all_sessions(self, features=FEATURES, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
        dfs = []
        for session in self.sessions:
            trial_dict = {}
            trial_dict['label'] = []
            for trial in session.trials:
                trial_dict['label'].append(trial.action_type)
                for channel in self.relevant_channel:
                    for band_name, (low, high) in FREQ_BANDS.items():
                        signal = trial.get_signal()[channel]
                        signal = bandpass_filter(signal, low, high, session.fs)
                        signal = hilbert(signal)
                        signal = np.abs(signal)
                        for feature in features:
                            n = len(signal)
                            for i, start in enumerate(range(0, n - window_size + 1, step_size)):
                                value = feature(signal[start:start + window_size])
                                label = f'CH{channel}_{band_name}_{feature.__name__}_window_{i}'
                                if label in trial_dict.keys():
                                    trial_dict[label].append(value)
                                else:
                                    trial_dict[label] = [value]          
            dfs.append(pd.DataFrame(trial_dict))
        return pd.concat(dfs)
    

class channel:
    def __init__(self, name, idx, location, participant, sessions):
        self.name = name
        self.idx = idx
        self.location = location
        self.participant = participant
        
        pds_baseline = []
        pds_aroundObjGrasped = []
        
        for session in sessions:
            for trial in session.trials:
                pds_baseline.append(trial.get_pds_baseline(idx))
                pds_aroundObjGrasped.append(trial.get_pds_aroundObjGrasped(idx))
                
        self.pds_baseline = np.mean(np.array(pds_baseline), axis=0)
        self.pds_aroundObjGrasped = np.mean(np.array(pds_aroundObjGrasped), axis=0)
        
        self.absolute_psd_difference = np.abs(self.pds_aroundObjGrasped - self.pds_baseline)
        self.corr = correlate(self.pds_baseline, self.pds_aroundObjGrasped)
        self.t_stat, self.p_value = ttest_rel(self.pds_baseline[1], self.pds_aroundObjGrasped[1])

    
class session: 
    def __init__(self, name, fs, neural_data, trials_info):
        self.name = name
        self.fs = fs
        self.neural_data = neural_data
        
        # TRIALS
        self.no_error = np.array(trials_info['ErrorCode']) == 0
        self.nb_trials = len(trials_info['TS_TrialStart'])
        self.trials = [trials(i, self.fs,  self.neural_data, trials_info) for i in range(self.nb_trials) if self.no_error[i]]
        
class trials: 
    def __init__(self, trial_idx, fs, neural_data, trials_info):
        trial_baseline_start = trials_info['TS_TrialStart'][trial_idx]
        trial_baseline_stop = trials_info['TS_CueOn'][trial_idx]
        self.trial_start = trials_info['TS_TrialStart'][trial_idx]
        trial_stop = trials_info['TS_HandBack'][trial_idx]
        
        self.trial_idx = trial_idx
        self.trials_info = trials_info
        self.fs = fs
        self.action_type = trials_info['ActionType'][trial_idx]
        
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
    
    def get_signal(self, trigger_start='TS_HandOut', trigger_stop='TS_HandBack', subsampling_frequency=SUBSAMPLING_FREQUENCY, baseline_correction=True, nb_samples=NB_SAMPLES):
        if trigger_start not in self.trials_info.keys() or trigger_stop not in self.trials_info.keys(): raise ValueError('Invalid trigger')
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
    
    def get_pds_baseline(self, channel_idx):
        return welch(self.baseline_signal[channel_idx], fs=self.fs)
    
    def get_pds_aroundObjGrasped(self, channel_idx):
        objectGrasp = self.trials_info['TS_ObjectGrasp'][self.trial_idx]
        start_idx = int((objectGrasp - 0.5) * self.fs) - int(self.trial_start * self.fs)
        stop_idx = int((objectGrasp + 0.5) * self.fs) - int(self.trial_start * self.fs)
        return welch(self.signal[channel_idx, start_idx:stop_idx], fs=self.fs)
    
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

    powergrid_noise_frequencies_Hz = [harmonic_idx*50 for harmonic_idx in range(1,nb_harmonics+1)] # removing 50Hz noise and its harmonics

    for noise_frequency in powergrid_noise_frequencies_Hz:
        sos = butter(N=4, Wn=(noise_frequency - 2, noise_frequency + 2), fs=fs, btype="bandstop", output="sos")
        signal = sosfiltfilt(sos, signal)
        
    return signal


if __name__ == '__main__':
    p = participant('s6')
    df = p.get_features_all_sessions()


