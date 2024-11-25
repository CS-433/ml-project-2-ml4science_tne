import pickle


def load_pickle(data_path):
    filename = data_path 
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
    
    return 0
