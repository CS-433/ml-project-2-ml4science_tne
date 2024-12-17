from utils import *
from dataset import *
from constants import *

random.seed(RANDOM_STATE)

def load_participant(part_name):
    """Load a participant from the saved directory.

    Args:
        part_name (str): Name of the participant.
    """
    participant = Participant(part_name)
    saved_dir = os.path.join(os.getcwd(), 'saved')
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    with open(f'saved/{part_name}.pkl', 'wb') as f:
        pickle.dump(participant, f, pickle.HIGHEST_PROTOCOL)    
    return participant
            
def load_features_mvt(participant):
    """Load movement features for a participant, for both execution and observation.

    Args:
        participant (Participant): Participant object.
    """
    
    ex_features = participant.get_features_all_sessions_mvt('E')
    ex_features.to_hdf(f'saved/ex_features_{part_name}_mvt.h5', 'df', mode='w', data_columns=True)
    
    ex_baseline_features = participant.get_features_all_sessions_unresponsive(movtype='E')
    ex_baseline_features.to_hdf(f'saved/ex_baseline_features_{part_name}_mvt.h5', 'df', mode='w', data_columns=True)
    
    obs_features = participant.get_features_all_sessions_mvt('O')
    obs_features.to_hdf(f'saved/obs_features_{part_name}_mvt.h5', 'df', mode='w', data_columns=True)
    
    obs_baseline_features = participant.get_features_all_sessions_unresponsive(movtype='O')
    obs_baseline_features.to_hdf(f'saved/obs_baseline_features_{part_name}_mvt.h5', 'df', mode='w', data_columns=True)
    
def load_features_ExObs(participant):  
    """Load features for a participant for action recognition.

    Args:
        participant (Participant): Participant object.
    """
    features = participant.get_features_all_sessions_ExObs()
    features.to_hdf(f'saved/features_{part_name}_ExObs.h5', 'df', mode='w', data_columns=True)
    
    baseline_features = participant.get_features_all_sessions_unresponsive(movtype=None)
    baseline_features.to_hdf(f'saved/baseline_features_{part_name}_ExObs.h5', 'df', mode='w', data_columns=True)
    
    
if __name__ == '__main__':
    
    for part_name in PARTICIPANTS:
        participant = load_participant(part_name)
        load_features_mvt(participant)
        load_features_ExObs(participant)

    