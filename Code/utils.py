import pickle

from scipy.signal import butter, sosfiltfilt

from constants import FS

def load_pickle(data_path):
    filename = data_path +  '.pickle'
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

    
def compute_acc(preds, y):
    pass

def compute_F1(preds, y):
    pass