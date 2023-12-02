import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import scipy.signal as signal 
import scipy
import scipy.ndimage as ndimage
import tqdm
import argparse
from pathlib import Path

# Remove peaks where the difference of two successive values is greater than threshold
# but only if the sign of the difference is opposite to the previous peak
def reject_outliers(data: np.ndarray, threshold=600):
    filt = data.copy()
    diff = np.diff(data, n=1)
    last_peak_i = None
    last_peak_pos_edge = None
    for i, v in enumerate(diff):
        if abs(v) > threshold:
            is_pos_edge = np.sign(v) > 0
            if last_peak_i is None or is_pos_edge == last_peak_pos_edge:
                last_peak_i = i
                last_peak_pos_edge = is_pos_edge
            elif is_pos_edge != last_peak_pos_edge:
                filt[last_peak_i:i+1] = data[last_peak_i-1]
                last_peak_i = None
        if i - (last_peak_i or 0) > 100:
            last_peak_i = None
    return filt

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--input", type=str, default="C:/Users/n3xed/OneDrive - OST/Hackathon 2023/fuellstandsensoren-glassammelstellen-gruenglas.csv")
parser.add_argument("--output", type=str, default="")

args = parser.parse_args()
filename = args.input
out_file = args.output if args.output else str(Path(filename).with_suffix("")) + "-processed.csv"

sensor_data = pd.read_csv(filename, delimiter=";")
sensor_data['measured_at'] = pd.to_datetime(sensor_data['measured_at'], utc=True)

rel_data = sensor_data[["name", "measured_at", "data_distance"]].sort_values(by="measured_at").copy()
rel_data["measured_at"] = rel_data["measured_at"].dt.tz_convert("Europe/Berlin").map(pd.Timestamp.timestamp)

rel_data = list(rel_data.groupby("name"))

res_names = []
res_dist = []
res_time = []
res_region = []

for name, data in tqdm.tqdm(rel_data):
    data_dist = np.array(data["data_distance"])
    data_time = np.array(data["measured_at"])
    
    for i in range(len(data_dist)):
        if np.isnan(data_dist[i]) or data_dist[i] == np.inf or data_dist[i] == -np.inf:
            data_dist[i] = data_dist[i-1] if not np.isnan(data_dist[i-1]) else 0.0
    
    data_dist_filt = reject_outliers(reject_outliers(data_dist))
    # data_dist_filt = signal.savgol_filter(data_dist_filt, 20, 3)
    data_dist_filt_diff = np.diff(data_dist_filt, n=1)
    data_dist_filt_gauss = ndimage.gaussian_filter1d(data_dist_filt, sigma=2)
    
    peaks, _ = signal.find_peaks(np.diff(data_dist_filt_gauss), prominence=60)
    peaks = [*peaks[1:], len(data_dist_filt)]

    time_regions = [data_time[peaks[n]+1:peaks[n+1]] for n in range(len(peaks)-1)]
    dist_regions = [ndimage.gaussian_filter1d(data_dist_filt[peaks[n]+1:peaks[n+1]], sigma=2) for n in range(len(peaks)-1)]

    for region_i in range(len(time_regions)):
        for t, d in zip(time_regions[region_i], dist_regions[region_i]):
            res_names.append(name)
            res_dist.append(d)
            res_time.append(t)
            res_region.append(region_i)

res_frame = pd.DataFrame({"name": res_names, "timestamp": res_time, "distance": res_dist, "period": res_region}, index=None)
print(res_frame)
res_frame.to_csv(out_file, index=False)