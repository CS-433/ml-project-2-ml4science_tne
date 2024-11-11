import numpy as np
import matplotlib.pyplot as plt
import pyedflib as pyedf
import pickle as pickle
import os, csv, glob, json, hdf5storage
from os import mkdir, path
from pyxdf import load_xdf



def save_variable(variable, name):
    final_name = name + '.pickle'
    with open(final_name, 'wb') as f:
        pickle.dump(variable, f, pickle.HIGHEST_PROTOCOL)
    return 0

        
        
def load_pickle_variable(variable_name):
    tmp = variable_name.split('.')
    if tmp[-1] != 'pkl': filename = variable_name +  '.pickle'
    else: filename = variable_name
    with open(filename, 'rb') as f:
        variable = pickle.load(f)
    return variable



def create_sub_folder(_path_, subname):
    sub_dir_name = _path_ + '/' + subname
    if not(path.exists(sub_dir_name)):
        os.mkdir(sub_dir_name)
    return sub_dir_name



def read_tsv_file(file_path):
    info = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as tsv_file:
#    with open(file_path, 'r', newline = '') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        # Skip the header row if needed
        header = next(reader)
        n_vars = len(header)
        all_data_info = []
        for row in reader:
            all_data_info.append(row)
        all_data_info = np.asarray(all_data_info)
        for var in range(n_vars):
            info[header[var]] = all_data_info[:,var]
    return info



def read_ecog_edf_file(file_path, verbose = 0):
    # Initializing data structure
    data_tmp = {'Data' : [], 'Labels' : []}
    try:
        # Reading the EDF file
        edf_file = pyedf.EdfReader(file_path)
        # Get the number of channels
        num_channels = edf_file.signals_in_file
        if verbose: print('There are %d channels.'%num_channels)
        # Get the signal labels (channel names)
        channel_labels = edf_file.getSignalLabels()
        data_tmp['Labels'] = np.asarray(channel_labels)
        # Get the signal samples for each channel
        for channel in range(num_channels):
            data_tmp['Data'].append(edf_file.readSignal(channel))
        data_tmp['Data'] = np.asarray(data_tmp['Data'])
    except Exception as e:
        print("Error:", e)
    return data_tmp



def read_JSON_file(filename):
    with open(filename, 'r') as f:
        # Load the JSON data
        data = json.load(f)
    return data



def remap_string_trials_to_code(info_trials, global_parameters):
    tmp_strings = info_trials['trial_type']
    new_vals = [global_parameters['Trial_map'][tmp_string] for tmp_string in tmp_strings]
    return new_vals



def load_all_data(list_subjs, global_parameters, verbose = 0):
    all_data = {}
    for subj in list_subjs:
        subj_folder = global_parameters['data_path'] + '%s'%subj
        if verbose: print('---------- Working on subject %s'%subj)
        all_data[subj] = {}
        sessions_folders = glob.glob(subj_folder + '/ses*')
        for session_folder in sessions_folders:
            session_folder = session_folder.replace('\\', '/')
            session = session_folder.split('/')[-1]
            if verbose: print('---- Session %s'%session)
            all_data[subj][session] = {'Exp_Data' : [], 
                                       'Channel_labels' : [],
                                       'Blocks' : [],
                                       'Trials_info' : [],
                                       'Filenames' : [],
                                       'Baseline' : {'Data' : [], 'Sess' : [], 'Runs' : [], 'Type' : []}
                                      }
            ################ ECOG AND EVENT FILES ################
            if verbose: print('Loading ECoG and Event files...')
            tmp_ecog_files = glob.glob(session_folder + '/ecog/%s_%s_task-mapping*.edf'%(subj, session))
            blocks_IDs = []
            sessions_IDs = []
            file_names = []
            start_triggers_EMG_sync = []
            for tmp_ecog_file in tmp_ecog_files:
                tmp_ecog_file = tmp_ecog_file.replace('\\' , '/')
                blocks_IDs.append(int(tmp_ecog_file.split('-')[-1][:2]))
                sessions_IDs.append(int(session.split('-')[-1]))
                data_tmp = read_ecog_edf_file(tmp_ecog_file)
                all_data[subj][session]['Exp_Data'].append(data_tmp['Data'])
                tmp_filename = tmp_ecog_file.split('/')[-1][:-8]
                file_names.append(tmp_filename)
                # Loading the corresponding event file (named the same but with different folder and final extension
                tmp_beh_file = session_folder + '/beh/' + tmp_filename + 'events.tsv'
                info_tmp = read_tsv_file(tmp_beh_file)
                info_tmp['trial_type'] = remap_string_trials_to_code(info_tmp, global_parameters)
                all_data[subj][session]['Trials_info'].append(info_tmp)
                # Loading the json file to get the start trigger in the case of sess 4 and 5
                tmp_json_file = session_folder + '/beh/' + tmp_filename + 'beh.json'
                json_data = read_JSON_file(tmp_json_file)
                try: 
                    start_trig = json_data['startTrig']
                    start_triggers_EMG_sync.append(start_trig)
                except: 
                    print('No startTrig info for session %s'%session)
            if len(start_triggers_EMG_sync) > 0: all_data[subj][session]['Start_triggers'] = np.asarray(start_triggers_EMG_sync)
            all_data[subj][session]['Filenames'].append(file_names)
            all_data[subj][session]['Channel_labels'].append(data_tmp['Labels'])
            all_data[subj][session]['Blocks'] = np.asarray(blocks_IDs)
            ################ ECOG BASELINE FILES ################
            if verbose: print('Loading Baseline files...')
            tmp_ecog_baseline_files = glob.glob(session_folder + '/ecog/%s_%s_task-baseline*.edf'%(subj, session))
            runs_baseline = []
            eyes_baseline = []
            sessions_baseline_IDs = []
            for tmp_ecog_baseline_file in tmp_ecog_baseline_files:
                runs_baseline.append(int(tmp_ecog_baseline_file.split('-')[-1][:2]))
                eyes_baseline.append(tmp_ecog_baseline_file.split('_')[-3][-2:])
                sessions_baseline_IDs.append(int(session.split('-')[-1]))
                tmp_baseline_data = read_ecog_edf_file(tmp_ecog_baseline_file)
                all_data[subj][session]['Baseline']['Data'].append(tmp_baseline_data['Data'])
            all_data[subj][session]['Baseline']['Runs'] = np.asarray(runs_baseline)
            all_data[subj][session]['Baseline']['Sess'] = np.asarray(sessions_baseline_IDs)
            all_data[subj][session]['Baseline']['Type'] = np.asarray(eyes_baseline)
    return all_data



def reorder_data_channels_order(all_data):
    for subj in all_data.keys():
        for sess in all_data[subj].keys():
            # The labels would always be in the same order across subjects and sessions.
            channels_n = [int(label[-2::]) for label in np.squeeze(all_data[subj][sess]['Channel_labels'])]
            ordered_ch_id = np.argsort(channels_n)
            all_data[subj][sess]['Channel_labels'] = np.squeeze(all_data[subj][sess]['Channel_labels'])[ordered_ch_id]
            # Reordering the channels inside every block of data
            n_blocks = len(all_data[subj][sess]['Exp_Data'])
            for block in range(n_blocks):
                tmp_data = all_data[subj][sess]['Exp_Data'][block] 
                all_data[subj][sess]['Exp_Data'][block] = tmp_data[ordered_ch_id]
    return all_data



def load_EMG_xdf_file(filename):
    file_emg, _ = load_xdf(filename)
    n_streams = len(file_emg)
    EMG_data = {'Data' : [], 'Info' : [], 'Time_stamps' : [], 'Triggers' : []}
    for stream in range(n_streams):
        tmp_stream_data = file_emg[stream]
        tmp_info = tmp_stream_data['info']
        tmp_time_series = tmp_stream_data['time_series']
        tmp_time_stamps = tmp_stream_data['time_stamps']
        if tmp_info['name'][0] ==  'Labjack':
            EMG_data['Data'] = np.squeeze(tmp_time_series)
            EMG_data['Info'] = tmp_info
            EMG_data['Time_stamps'] = np.squeeze(tmp_time_stamps)
        if tmp_info['name'][0] ==  'Trials_triggers':   
            EMG_data['Triggers'] = np.squeeze(tmp_time_stamps)
    return EMG_data



def load_EMG_data(neuro_data, list_subjs, global_parameters, verbose = 0):
    emg_data = {}
    subjs_folders = glob.glob('../../Data/sub*')
    for subj in list_subjs:
        emg_data[subj] = {}
        subj_folder = global_parameters['data_path'] + '%s'%subj
        sessions_folders = glob.glob(subj_folder + '/ses*')
        for session_folder in sessions_folders:
            session = session_folder.replace('\\', '/').split('/')[-1]
            print('----------------- %s'%session)
            ecog_filenames = np.squeeze(neuro_data[subj][session]['Filenames'])
            emg_data[subj][session] = {'EMG_data' : [], 
                                       'Channel_idx' : [0,1,2,3,5,6],
                                       'Muscles_names' : ['Extensor digitorum', 'Flexor digitorum', 'Biceps brachii', 'Triceps brachii', 'Deltoideus lateralis', 'Trapezius'],
                                       'Muscles_labels' : ['EXTD_R', 'FLED_R', 'BIC_R', 'TRI_R', 'DEL_R', 'TRA_R']
                                       }
            folder_path = session_folder.replace('\\','/') + '/emg' 
            extension = '.xdf'
            try: 
                for e in range(len(ecog_filenames)):
                    tmp_ecog_filename_split = ecog_filenames[e].split('mapping')
                    emg_file = 'block_emg_' + tmp_ecog_filename_split[-1] + '.xdf'
                    if verbose: print('%d - ECoG filename : %s'%(e + 1, ecog_filenames[e]))
                    if verbose: print('%d - Loading EMG : %s'%(e + 1, emg_file))
                    filename = folder_path + '/%s'%emg_file
                    tmp = load_EMG_xdf_file(filename)
                    tmp['Data'] = 1000*tmp['Data'][:,emg_data[subj][session]['Channel_idx']].T#To make it in mV
                    emg_data[subj][session]['EMG_data'].append(tmp)
            except: 
                print('Error when loading emg data.')
                del emg_data[subj][session]
    return emg_data



def merge_data(neuro_data, emg_data):
    for subj in emg_data.keys():
        for sess in emg_data[subj].keys():
            neuro_data[subj][sess]['EMG'] = emg_data[subj][sess]
    return neuro_data



def get_time_vector(n_tps, global_parameters):
    fs = global_parameters['fs']
    time_vec = np.linspace(0, n_tps/fs, n_tps)
    return time_vec



def fint_t_id(time_pts, t_sec):
    t_id = np.argmin(np.abs(time_pts - t_sec))
    return t_id



def get_key_by_value(dictionary, value):
    for key, values in dictionary.items():
        for val in values:
            if val == value:
                return key
    return None



def build_symmetric_matrix(upper_diagonal_vector):
    n = int(1 + np.sqrt(1 + 8 * len(upper_diagonal_vector))) // 2 
    matrix = np.zeros((n, n))
    # Fill upper diagonal
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            matrix[i, j] = upper_diagonal_vector[idx]
            matrix[j, i] = upper_diagonal_vector[idx]  # Fill the corresponding lower diagonal element
            idx += 1
    return matrix



def create_checkerboard_matrix(vector):
    size = int(np.ceil(np.sqrt(len(vector) * 2)))
    n_per_row = int(len(vector)/size)
    matrix = -1*np.ones((size, size), dtype=vector.dtype)
    counter = 0
    for i in range(size):
        if i%2 == 1: matrix[i,::2] = vector[counter:counter + n_per_row]
        if i%2 == 0: matrix[i,1::2] = vector[counter:counter + n_per_row]
        counter += n_per_row
    return matrix.T



def create_full_matrix_contributions_per_region(contributions, ordered_ch_labels_ROI, ordered_ch_labels):
    new_contrs_all_bands = []
    for b in range(contributions.shape[0]):
        tmp_contr = contributions[b]
        new_list_contrs = []
        for i in range(len(ordered_ch_labels)):
            if ordered_ch_labels[i] not in ordered_ch_labels_ROI:
                new_list_contrs.append(0)
            else:
                tmp_idx = np.where(ordered_ch_labels_ROI == ordered_ch_labels[i])[0][0]
                new_list_contrs.append(tmp_contr[tmp_idx])
        new_contrs_all_bands.append(new_list_contrs)
    new_contrs_all_bands = np.asarray(new_contrs_all_bands)
    return new_contrs_all_bands



