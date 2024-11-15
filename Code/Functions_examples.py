import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd 
import pickle
import os
import h5py
import glob


def create_sub_folder(_path_, subname):
    sub_dir_name = _path_ + '/' + subname
    if not(os.path.exists(sub_dir_name)):
        os.mkdir(sub_dir_name)
    return sub_dir_name


def save_variable(variable, name):
    final_name = name + '.pickle'
    with open(final_name, 'wb') as f:
        pickle.dump(variable, f, pickle.HIGHEST_PROTOCOL)

        
def load_pickle_variable(variable_name):
    filename = variable_name +  '.pickle'
    with open(filename, 'rb') as f:
        variable = pickle.load(f)
    return variable


def select_trials(data_subj, cond_dict, verbose = 0):
    # data_subj contains the data of a specific subject after having already concatenated the trials of different sessions (which means that the epoching)
    # has already been done.
    # Example of cond_dict (which is the dictionary containing the info of the trials you want to extract)
    # cond_dict = {'action_type' : 'E', 'cue_type' : 'C', 'object_size' : 'B', 'object_color' : '', 'object_position' : '', 'correct' : 1}
    # You can leave object_color and object_position blank, since there are not factor of interest for us. 
    # action_type: E = Execution, O = Observation
    # cue_type: C = Cues, M = mimic (choose ONLY C)
    # object_size: B = Big (power grasp), S = Small (precision grip)
    # So, for example, if you choose to select the trials EC, you are selecting all the executed trials. If you select ECS, you are selecting all the precision
    # grip executed trials
    action_type = cond_dict['action_type']
    cue_type = cond_dict['cue_type']
    object_size = cond_dict['object_size']
    object_color = cond_dict['object_color']
    object_position = cond_dict['object_position']
    correct_flag = cond_dict['correct']
    desired_code = action_type + cue_type + object_size + object_color + object_position
    # Extracting the trials info 
    trials_info_tmp = data_subj['trials_info']
    # There is no ObjectGrasped for the imagination trials (it is assigned to NaN values)
    n_trials = len(trials_info_tmp['CorrectGrasped'])
    if action_type in ['E', 'O']:
        correct_flags = np.asarray(trials_info_tmp['CorrectGrasped'])
    elif action_type == 'I':
        correct_flags = np.asarray([1]*n_trials)
    error_flags =  np.asarray(trials_info_tmp['ErrorCode'])
    action_type_flags = np.asarray(trials_info_tmp['ActionType'])
    cue_type_flags = np.asarray(trials_info_tmp['CueType'])
    object_size_flags = np.asarray(trials_info_tmp['ObjectSize'])
    object_color_flags = np.asarray(trials_info_tmp['ObjectColor'])
    object_position_flags = np.asarray(trials_info_tmp['ObjectPosition'])
    sequence_match_flags = np.asarray(trials_info_tmp['SequenceMatch'])
    try: 
        mistakes_catched_flags = np.asarray(trials_info_tmp['CatchMistakeDetected'])
    except: 
        print('Not possible to find the info related to CatchMistakeDetected')
        mistakes_catched_flags = np.asarray([-1]*n_trials)
    # Identifying trials corresponding for action type
    if action_type != '': action_type_bools = action_type_flags == action_type
    else: action_type_bools = [True]*n_trials
    if verbose: print('N. Trials per Action type : ',np.sum(action_type_bools))
    # Identifying trials corresponding for cue type   
    if cue_type != '': cue_type_bools = cue_type_flags == cue_type
    else: cue_type_bools = [True]*n_trials
    if verbose: print('N. Trials per Cue type : ',np.sum(cue_type_bools))
    # Identifying trials corresponding for object size 
    if object_size != '': object_size_bools = object_size_flags == object_size
    else: object_size_bools = [True]*n_trials
    if verbose: print('N. Trials per Object size : ',np.sum(object_size_bools))
    # Identifying trials corresponding for object color  
    if object_color != '': object_color_bools = object_color_flags == object_color
    else: object_color_bools = [True]*n_trials
    if verbose: print('N. Trials per Object color : ',np.sum(object_color_bools))
    # Identifying trials corresponding for object position
    if object_position != '': object_position_bools = object_position_flags == object_position
    else: object_position_bools = [True]*n_trials
    if verbose: print('N. Trials per Object position : ',np.sum(object_position_bools))
    # Selecting trials based on correct or not
    if correct_flag:
        correct_grasp_bools = correct_flags == 1
        no_error_bools = error_flags == 0
        mistakes_catched_bools = mistakes_catched_flags == -1
        correct_bools = correct_grasp_bools * no_error_bools * mistakes_catched_bools
    else: correct_bools = [True]*n_trials
    if verbose: print('N. Trials per Correct trials : ',np.sum(correct_bools))
    # Sequence Match bools
    sequence_match_bools = sequence_match_flags == 1
    if verbose: print('N. Trials not following the correct Sequence : ',np.sum(sequence_match_flags == 0))
    bools_trials_desired = action_type_bools * cue_type_bools * object_size_bools * object_color_bools * object_position_bools * correct_bools * sequence_match_bools
    idx_trials_desired = np.where(bools_trials_desired == 1)[0]
    if verbose:
        print('There are %d trials satisfying the desired trial code %s.'%(len(idx_trials_desired), desired_code))
    selected_data = {}
    for trigger in data_subj['neural_data'].keys():
        selected_data[trigger] = data_subj['neural_data'][trigger][idx_trials_desired, :, :]
    return selected_data, idx_trials_desired, desired_code


def epoch_trials_precise_segments(data, trigger1 = 'TS_HandOut', trigger2 = 'TS_ObjectGrasp', verbose = 0):
    # This functions has the goal of extracting for each trial the precise segment between two specified triggers (this means that every trial will have a 
    # different duration)
    epoched_data = {}
    for subj in data.keys():
        if verbose:
            print('-------- Epoching subj %s --------'%subj)
        epoched_data[subj] = {}
        for sess in data[subj].keys():
            if verbose:
                print('---- %s '%sess)
            epoched_data[subj][sess] = {}
            info_trials_tmp = data[subj][sess]['trials_info']
            epoched_data[subj][sess]['trials_info'] = info_trials_tmp
            fs_tmp = data[subj][sess]['fs']
            epoched_data[subj][sess]['fs'] = fs_tmp
            epoched_data[subj][sess]['channel_labels'] = data[subj][sess]['channel_labels']
            epoched_data[subj][sess]['channel_locations'] = data[subj][sess]['channel_locations']
            epoched_data[subj][sess]['neural_data'] = {}
            epoched_data[subj][sess]['timestamps'] = {}
            data_tmp = data[subj][sess]['neural_data']
            n_channels, n_ts = np.shape(data_tmp)
            ts = np.arange(n_ts)/fs_tmp
            n_trials = len(info_trials_tmp['TrialID'])
            TS_tmp1 = info_trials_tmp[trigger1]
            TS_tmp2 = info_trials_tmp[trigger2]
            all_trials = []
            all_ts = []
            for trial in range(n_trials):
                if TS_tmp1[trial] != 0 and TS_tmp2[trial] != 0:
                    id_trig1 = find_timestamp_id(ts, TS_tmp1[trial])
                    id_trig2 = find_timestamp_id(ts, TS_tmp2[trial])
                    trial_tmp = data_tmp[:, id_trig1 : id_trig2]
                    all_trials.append(trial_tmp)
                    all_ts.append(np.linspace(0, ts[id_trig2] - ts[id_trig1], id_trig2 - id_trig1))
                else:
                    # If the trial failed before this trigger was sent
                    all_trials.append(np.nan)
                    all_ts.append(np.nan)
            epoched_data[subj][sess]['neural_data'][trigger1] = all_trials
            epoched_data[subj][sess]['timestamps'][trigger1] = all_ts
    return epoched_data


def find_timestamp_id(timestamps, t):
    return np.argmin(np.abs(timestamps - t))