import os
import sys

import numpy as np
import pandas as pd

def test_separate_frequency_bands():
    df = pd.DataFrame(
        {
            'participant': ['A', 'A', 'B', 'B'],
            'session': ['S1', 'S2', 'S1', 'S1'],
            'condition': ['obs', 'ex', 'obs', 'ex'],
            
            'neural_data': [
                # Delta and Gamma bands
                [[np.cos(2*np.pi*2*x) for x in np.linspace(0, 10, 1000)], [np.cos(2*np.pi*40*x) for x in range(1000)]],
                # Theta band
                [[np.cos(2*np.pi*5*x) for x in np.linspace(0, 10, 1000)]],
                # Alpha band
                [[np.cos(2*np.pi*10*x) for x in np.linspace(0, 10, 1000)]],
                # Beta band
                [[np.cos(2*np.pi*15*x) for x in np.linspace(0, 10, 1000)]]
            ]
        }
    )
    
    print(df)
    df_sep = separate_frequency_bands(df)
    print(df_sep[['channel0_Delta', 'channel0_Theta', 'channel0_Alpha', 'channel0_Beta', 'channel0_Beta', 'channel0_Gamma']])
    
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '../Code'))
    from dataset import separate_frequency_bands
    
    test_separate_frequency_bands()