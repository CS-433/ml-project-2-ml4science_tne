import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

def test_separate_frequency_bands():
    t = np.linspace(0, 1, 2028, endpoint=False)
    df = pd.DataFrame(
        {
            'participant': ['A', 'A', 'B', 'B'],
            'session': ['S1', 'S2', 'S1', 'S1'],
            'condition': ['obs', 'ex', 'obs', 'ex'],
            
            'neural_data':[
                # Delta and Gamma bands
                np.array([np.sin(2*np.pi*2*t), np.sin(2*np.pi*40*t)]),
                # Theta band
                np.array([np.sin(2*np.pi*5*t)]),
                # Alpha band
                np.array([np.sin(2*np.pi*10*t)]),
                # Beta band
                np.array([np.sin(2*np.pi*15*t)])
            ]
        }
    )
    
    print(df)
    df_sep = separate_frequency_bands(df)
    print(df_sep[['channel0_Delta', 'channel0_Theta', 'channel0_Alpha', 'channel0_Beta', 'channel0_Gamma']])
    
def test_remove_electrical_noise():
    data = {
        's1': {
            'sess1': {
                'neural_data': np.array([[np.cos(2*np.pi*50*x) for x in np.linspace(0, 10, 20480)],
                                [np.cos(2*np.pi*25*x) for x in np.linspace(0, 10, 20480)]]),
            }
        }
    }
    
    data_no_elec_noise = remove_electrical_noise(data)
    plt.plot(welch(data_no_elec_noise['s1']['sess1']['neural_data'][0], fs=2048))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Spectral Analysis of Neural Data')
    plt.show()
    
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '../Code'))
    from dataset import separate_frequency_bands, remove_electrical_noise
    
    test_separate_frequency_bands()
    # test_remove_electrical_noise()