import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from Utilities_and_Loading import *
import pickle
import os


############################################################################################################
# GENERAL FUNCTIONS
############################################################################################################

def convert_time_to_timepoint(time):
    """
    Convert a given time in milliseconds to a corresponding timepoint index.

    This function converts a specified time in milliseconds to a timepoint index based on a total 
    duration of 4000 milliseconds and a total of 2340 timepoints.

    Parameters:
    time (float): The time in milliseconds to be converted.

    Returns:
    int: The corresponding timepoint index.
    """
    total_time = 4000 # ms
    total_timepoints = 2340 
    return int(np.ceil(time/total_time*total_timepoints))


def save_dataset(trainingset, testset, filename, description) :
    """
    Save the training and test datasets to a specified file.

    This function saves the provided training and test datasets into a pickle file 
    located in the 'Data' directory one level above the current working directory.

    Parameters:
    trainingset (DataFrame): The training dataset to be saved.
    testset (DataFrame): The test dataset to be saved.
    filename (str): The name of the file to save the datasets.
    description (str): A description of the datasets.

    Returns:
    None
    """
    path = os.path.join(os.getcwd(), os.pardir, "Data", filename)
    data = {'trainingset': trainingset, 'testset': testset}
    pickle.dump({'datasets':data, 'description': description}, open(path, 'wb'))
    return


def load_dataset(filename) :
    """
    Load the training and test datasets from a specified file.

    This function loads the training and test datasets from a pickle file located in the 
    'Data' directory one level above the current working directory. It separates the 
    'Trials_type' column as the target variable for both training and test sets.

    Parameters:
    filename (str): The name of the file to load the datasets from.

    Returns: (tuple)
    tuple: A tuple containing the training set, test set, training labels (y_train), and test labels (y_test).
    str : A description of the datasets.
    """
    path = os.path.join(os.getcwd(), os.pardir, "Data", filename)
    dict = pickle.load(open(path, 'rb'))
    description = dict['description']
    data = dict['datasets']
    y_train = data['trainingset'].pop('Trials_type')
    y_test = data['testset'].pop('Trials_type')
    return data['trainingset'], data['testset'], y_train, y_test, description

def save_results_nestedcrossval(filename, model, predictions, balanced_accuracy, balanced_accuracy_scores, confusion_matrix, chance_level, chance_level_scores, dataset_name, param_grid, grid_search_memory, notStratified, bySession, random_state = np.NaN, k=np.NaN):
    
    path_name = os.path.join(os.getcwd(), os.pardir, "Results", filename + '.pkl')
    
    # Avoid overwriting file if error in the name
    if os.path.exists(path_name):
        path_name = os.path.join(os.getcwd(), os.pardir, "Results", filename + '_' + datetime.today().strftime('%Y-%m-%d') + '.pkl')
        
        
    dict = {'model': model, 
            'predictions': predictions, 
            'balanced_accuracy': balanced_accuracy, 
            'balanced_accuracy_scores': balanced_accuracy_scores,
            'chance_level': chance_level,
            'chance_level_score': chance_level_scores,
            'confusion_matrix': confusion_matrix,
            'dataset_name': dataset_name,
            'param_grid': param_grid,
            'grid_search_memory': grid_search_memory,
            'random_state': random_state,
            'k': k,
            'notStratified': notStratified,
            'bySession': bySession
            }
    
    pickle.dump(dict, open(path_name, 'wb'))
    return None


def load_results(filename):
    path_name = os.path.join(os.getcwd(), os.pardir, "Results", filename + '.pkl')
    dict = pickle.load(open(path_name, 'rb'))
    return dict

def shuffle_column(y, n_shuffles=1000, random_state=42):
    # create 1000 shuffled versions of the labels
    np.random.seed(random_state)
    shuffled_y = [np.random.permutation(y) for _ in range(n_shuffles)]
    return np.array(shuffled_y)


############################################################################################################
# GET STARTED 
############################################################################################################

def get_data(timebin = 100):
    """
    Load, preprocess, and split data into training and test sets.

    This function loads preprocessed data from a specified path and removes the keys 'Ch_labels' and 'Ch_locations' 
    with different dimensions. It flattens multi-dimensional array data by averaging over specified time bins, 
    then converts the data into a pandas DataFrame. Trials with movement types 'RE-E' and 'WE-E' are removed, and a 
    stratified split is performed to maintain the same distribution of sessions in the training and test sets. 
    The function removes session and filename labels and maps movement categories to numerical values: 'RE-I' to 0, 
    'WE-I' to 1, 'WL-I' to 2, and 'GR-I' to 3. Finally, the data is split into training and test sets based on the 
    stratified split indices.

    Parameters:
    timebin (int): The size of the time bin for averaging data. Default is 100.

    Returns:
    trainingset, testset (tuple of pandas DataFrames): A tuple containing the training set and the test set.
    """
    # Load data
    data_path = r'C:\Users\sarah\epfl.ch\Leonardo Pollina - Sarah_Mikami\Data'
    data = load_pickle_variable(data_path + '/Data_single_tasks_preprocessed')
    
    # Remove the following keys : Ch_labels, and Ch_locations (not the same dimension)
    data2 = data.copy()
    keys_to_remove = ['Ch_labels', 'Ch_locations']
    for key in keys_to_remove:
        data2.pop(key, None)
    
    # Initialize an empty dictionary to hold the flattened data
    flattened_data = {}

    # Loop through the data dictionary
    for key, value in data2.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            for i in range(value.shape[1]):
                for j in range(value.shape[2]) :
                    if np.mod(j, timebin) == 0:
                        if j > value.shape[2] - timebin :
                            flattened_data[f'{key}_{i+1}_t{j}'] = np.mean(value[:, i, j:], axis=1) # Handle incomplete timebins, include as it is 
                        else:
                            flattened_data[f'{key}_{i+1}_t{j}'] = np.mean(value[:, i, j: j+timebin], axis=1) 
        else:
            # If the value is not a multi-dimensional array, just add it as is
            flattened_data[key] = value
    
    
    # Convert to dataframe 
    df = pd.DataFrame(flattened_data)
    
    # Remove the exectuted movements trials
    df = df[~df['Trials_type'].isin(['RE-E', 'WE-E'])]
    
    # Stratified split (keep the same distribution of sessions)
    session_labels = df['Session']
    n_trials = len(df['Session'])
    train_indices, test_indices = train_test_split(
        np.arange(n_trials),
        test_size=0.2,  # 20% test size
        stratify=session_labels,
        random_state=42
    )
    
    # Remove the session labels and the filenames labels 
    df = df.drop(columns=['Session', 'Filenames'])
    
    # Mapping of the categories of movements (Trials_type)
    mapping = {'RE-I': 0, 'WE-I': 1, 'WL-I': 2, 'GR-I': 3}
    df['Trials_type'] = df['Trials_type'].map(mapping)
        
    # Split the data into training and test sets
    trainingset = df.iloc[train_indices]
    testset = df.iloc[test_indices]
    
    
    return trainingset, testset


