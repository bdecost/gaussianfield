import os
import numpy as np
import pandas as pd

def load_VNbO2():
    """ load VNbO2 metadata into pandas, and XRD data into numpy (via pandas)
    Combiview metadata comes in multiple files
    join on the same index across all these files...
    """
    
    metadata = [
        # 'VO2 - Nb2O3 Composition Combiview.txt',
        'VO2 - Nb2O3 Phase Labels Combiview.txt',
        'VO2 - Nb2O3 Composition and temp Combiview.txt',
        'VO2 - Nb2O3 Position Combiview.txt'
    ]
    metadata = [os.path.join(os.path.dirname(__file__), os.pardir, 'data', filename) for filename in metadata]

    df = pd.concat(
        list(map(lambda f: pd.read_csv(f, sep='\t'), metadata)),
        axis=1
    )

    xrd = pd.read_csv(os.path.join(os.pardir, 'data', 'VO2 -Nb2O3 XRD Combiview.txt'), sep='\t')
    X = np.array(xrd)
    Xnorm = X / np.linalg.norm(X, ord=1, axis=-1)[:,np.newaxis]
    angle = np.array(xrd.keys(), dtype=float)

    # use zero-indexed labels...
    df['Label'] = df['Label'] - 1

    return X, angle, df
