import numpy as np
import statsmodels.api as sm
import pandas as pd

# Calculate the residual standard error (in days) and R^2 value for a given slope
# and test data
def get_accuracy(slope: float, periods: pd.DataFrame):
    times = [(p[0], np.array(p)) for p in periods["timestamp"]]
    times = np.hstack([p - t0 for t0, p in times]) # x variable
    fullnesses = np.hstack(periods["distance"].to_numpy()) # y variable
    
    rss = np.sum(((times - (fullnesses / slope)) / (24 * 3600)) ** 2)
    
    p_minus_two = len(times) - 2
    if p_minus_two <= 0:
        p_minus_two = 1
    rse = np.sqrt(rss / p_minus_two)
    
    rss = np.sum((fullnesses - (slope * times)) ** 2)
    tss = np.sum((fullnesses - np.mean(fullnesses)) ** 2)
    if tss == 0:
        r_squared = 0
    else:
        r_squared = 1 - (rss / tss)
    return rse, r_squared