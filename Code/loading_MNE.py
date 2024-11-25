import numpy as np
import pandas as pd
import mne 
import os
import scipy.signal as signal
from constants import DATA_PATH, FIF_PATH, SUBSAMPLING_FREQUENCY
from utils import load_pickle


def create_fif_file(data_path=DATA_PATH, verbose=False):
    
    if verbose : print(f'loading data')
    data = load_pickle(data_path)
    
    # channel_locations = get_highest_number_of_channels(data)
    # nb_channels = len(channel_locations)
    # channel_labels = [f"Ch{i+1}" for i in range(nb_channels)]
    
    
    for participant in data.keys():
        all_epochs = []
        for session in data[participant].keys():
            if verbose : print(f'participant: {participant}, session: {session}')
            epochs = create_epochs(data, participant, session, verbose=verbose)
            all_epochs.append(epochs)
    
        if verbose : print(f'start to concatenate epochs')
        combined_epochs = mne.concatenate_epochs(all_epochs)
        path = os.path.join(os.getcwd(), 'Data', f'Dataset_subjects_{participant}_Exe_Obs.fif')
        combined_epochs.save(path, overwrite=True)
    
    return None 
    
def create_epochs(data, participant, session, verbose):
    
    trials_NoError = np.array(data[participant][session]['trials_info']['ErrorType']) == 'NoError'
    
    neural_data = data[participant][session]['neural_data']
    fs = data[participant][session]['fs']
    trials_info = {
        'ParticipantID': np.array(data[participant][session]['trials_info']['ParticipantID'])[trials_NoError],
        'SessionID': np.array(data[participant][session]['trials_info']['SessionID'])[trials_NoError],
        'ActionType': np.array(data[participant][session]['trials_info']['ActionType'])[trials_NoError],
    }
    channel_locations = data[participant][session]['channel_locations'][0]
    nb_channels = len(channel_locations)
    channel_labels = [f"Ch{i+1}" for i in range(nb_channels)]
    
    
    # missing_channels, missing_channels_idx = get_missing_channels(data, participant, session, channel_locations)
    # for i, loc in enumerate(missing_channels):
    #     if verbose : print(f'adding {i} missing channels / {len(missing_channels)}')
    #     neural_data = np.insert(neural_data, missing_channels_idx[i], None, axis=0)
        
      
    # For each trial
    t_start = np.array(data[participant][session]['trials_info']['TS_TrialStart'])[trials_NoError]
    t_CueOn = np.array(data[participant][session]['trials_info']['TS_CueOn'])[trials_NoError]
    t_HandBack = np.array(data[participant][session]['trials_info']['TS_HandBack'])[trials_NoError]
    
    # Max duration 
    tmax = np.max(t_HandBack)
    
    # Timeline 
    timeline = np.arange(0, tmax+1/fs, 1/fs)
    
    # Duration
    duration_baseline = np.unique(data[participant][session]['trials_info']['inBaseDuration'])[0] / 1000 # ms converted to s
    std_duration = 7*duration_baseline
    
    nb_samples = int(std_duration * fs)
    nb_samples_baseline = int(duration_baseline * fs)
    
    # Numer of trial
    nb_trials = len(t_start)  
    
    data = np.zeros((nb_trials, nb_channels, nb_samples + nb_samples_baseline))
    for trial_idx in range(nb_trials):
        if verbose : print(f'trial: {trial_idx} / {nb_trials}')
        # time 
        tstart = t_start[trial_idx]
        tCueOn = t_CueOn[trial_idx]
        tHandBack = t_HandBack[trial_idx]
        
        # indices
        tstart_idx = np.where(np.isclose(timeline, tstart))[0][0]
        tCueOn_idx = np.where(timeline == tCueOn)[0][0]
        tHandBack_idx = np.where(timeline == tHandBack)[0][0]
        
        baseline = signal.resample(neural_data[:,tstart_idx:tCueOn_idx], nb_samples_baseline, axis=1)
        exp_signal = signal.resample(neural_data[:,tCueOn_idx:tHandBack_idx], nb_samples, axis=1)
        data[trial_idx] = np.concatenate((baseline, exp_signal), axis=1)
    
    tmin = -duration_baseline
    
    # Create MNE Info object
    info = mne.create_info(ch_names=channel_labels, sfreq=fs, ch_types='eeg')
    
    # Baseline
    baseline = (-nb_samples_baseline, 0)
    
    epochs = mne.EpochsArray(data, info, tmin=nb_samples_baseline+1, baseline=baseline)
    
    # Convert trials_info to a DataFrame
    metadata = pd.DataFrame(trials_info)

    # Attach metadata to epochs
    epochs.metadata = metadata
    
    return epochs


def get_highest_number_of_channels(data):
    
    channel_labels = []
    
    for participant in data.keys():
        for session in data[participant].keys():
            tmp_ch_locs = data[participant][session]['channel_locations'][0]
            unique_locs, counts_locs = np.unique(tmp_ch_locs, return_counts = True)
            
            for l, loc in enumerate(unique_locs):
                count = channel_labels.count(loc)
                if counts_locs[l] != count:
                    add = counts_locs[l] - count
                    for i in range(add):
                        channel_labels.append(loc)
    
    return np.array(channel_labels)


def get_missing_channels(data, participant, session, channel_locations):
    
    channel_locations = channel_locations.copy()
    channel_part_sess = data[participant][session]['channel_locations'][0]
    missing_channels = []
    missing_channels_idx = []
    
    for i, loc in enumerate(channel_locations):
        if loc in channel_part_sess:
            j = np.where(channel_part_sess == loc)[0][0]
            channel_part_sess = np.delete(channel_part_sess, j)
        else:
            missing_channels.append(loc)
            missing_channels_idx.append(i)
    
    return np.array(missing_channels), np.array(missing_channels_idx)


def main():
    create_fif_file(verbose=True)

if __name__ == "__main__":
    main()



