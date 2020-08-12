import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    # rolling window
    y_mean = y.rolling(
        window = int(rate/10), # desetina sekunde
        min_periods=1,
        center=True
    ).mean()

    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask