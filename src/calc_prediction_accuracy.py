import numpy as np
import statsmodels.api as sm
import pandas as pd

def get_accuracy(slope: float, periods: pd.DataFrame):
    times = [(p[0], p[-1]) for p in periods["timestamp"]]
    times = np.array([te - t0 for t0, te in times]) # x variable
    fullnesses = np.array([p[-1] for p in periods["distance"]]) # y variable
    
    rss = np.sum((times - (fullnesses / slope)) ** 2)
    rse = np.sqrt(rss / (len(times) - 2)) / (24 * 60 * 60)
    
    tss = np.sum((times - np.mean(times)) ** 2)
    r_squared = 1 - (rss / tss)

    return rse, r_squared