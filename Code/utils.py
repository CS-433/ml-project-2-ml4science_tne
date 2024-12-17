import pickle

def load_pickle(data_path):
    filename = data_path +  '.pickle'
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset
