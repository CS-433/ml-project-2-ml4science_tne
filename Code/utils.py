import pickle

def load_pickle(data_path):
    """Load a pickle dataset file.

    Args:
        data_path (str): Path to the pickle file.

    Returns:
        dictionary: Dataset.
    """
    filename = data_path +  '.pickle'
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset
