import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from csv import DictWriter
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import sem
import pandas as pd
import matplotlib.pyplot as plt

from calc_prediction_accuracy import get_accuracy


def plot_result(model, title):
    plt.plot(model)
    plt.title(title)
    plt.show()

def model_lin_reg(time, distance):
    # time_numeric = time # Converting nanoseconds to seconds
    # Convert the segments to NumPy arrays and reshape
    x_segment = np.array(time).reshape(-1, 1)
    y_segment = np.array(distance).reshape(-1, 1)
    # print(x_segment)
    # print(y_segment)

    model = LinearRegression()
    model.fit(x_segment, y_segment)
    slope = (model.coef_[0])

    ypred = model.predict(x_segment)
    mse = mean_squared_error(distance, ypred)
    # print(mse)
    return slope, mse

def calculate_testing(mean_slope, testing_data):
    print(testing_data)
    print(type(testing_data))

# Load the data from the uploaded CSV file
# As the delimiter is specified as ';', we'll use that to read the CSV
basepath = Path("C:/Users/cedri/OneDrive - OST/Hackathon 2023/")

weisglas_path = basepath / 'fuellstandsensoren-glassammelstellen-weissglas-processed.csv'
braunglas_path = basepath / 'fuellstandsensoren-glassammelstellen-braunglas-processed.csv'
grunglas_path = basepath / 'fuellstandsensoren-glassammelstellen-gruenglas-processed.csv'


for glass_type in ["weisglas","braunglas","grunglas"]:
    if glass_type == "weisglas":
        path = weisglas_path
    elif glass_type == "braunglas":
        path = braunglas_path
    elif glass_type == "grunglas":
        path = grunglas_path
    else:
        raise NotImplementedError()
    data = pd.read_csv(path, delimiter=',')
    data = data.sort_values(by='timestamp')
    counter = 0
    result_mse = {}
    result_slope = {}
    result_slope_list = []
    test_datas = pd.DataFrame()
    for group, collection_point in data.groupby("name"):
        model = LinearRegression()
        group_min = collection_point["distance"].min()
        group_max = collection_point["distance"].max()
        collection_point["distance"] = collection_point["distance"].apply(lambda v: 1-v/group_max)
        combined_data = collection_point.groupby("period")[["timestamp","distance"]].agg(list)
        training_data, test_data = train_test_split(combined_data, test_size=0.2, shuffle=False)
        test_datas = pd.concat([test_datas, test_data], ignore_index=True)
        mse_list = []
        slope_list = []
        for index, period_data in training_data.iterrows():
            slope, mse = model_lin_reg(period_data["timestamp"], period_data["distance"])
            slope_list.append(slope)
            mse_list.append(mse)
    
        mean_slope =np.array(slope_list).mean()
        mean_mse = np.array(mse_list).mean()
        result_mse[group] = mean_mse
        result_slope[group] = mean_slope
        
        result_slope_list.append({"name": group, "slope": mean_slope})
        counter += 1
    
    rse, rsq = get_accuracy(mean_slope, test_datas)
    print(f"RSE: {rse}")
    print(f"RSQ: {rsq}")
    print(f"MSE: {result_mse}")
    with open(f"{glass_type}_slopes.csv", "w", newline="") as w:
        dict_writer = DictWriter(w, ["name","slope"])
        dict_writer.writeheader()
        dict_writer.writerows(result_slope_list)
        