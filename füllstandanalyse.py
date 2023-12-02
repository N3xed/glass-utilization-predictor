import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sys



def plot_data(x, y, name):
    plt.title(name)
    plt.plot(x, y) 
    plt.show()





# filename = "sample_fuellstandsensoren-glassammelstellen-weissglas.csv"
filename = "fuellstandsensoren-glassammelstellen-weissglas.csv"

sensor_data = pd.read_csv(filename, delimiter=";")
sensor_data['measured_at'] = pd.to_datetime(sensor_data['measured_at'])
first_year = sensor_data[(sensor_data['measured_at'] >= pd.Timestamp(2020,7,1, tz="Europe/Amsterdam")) & (sensor_data["measured_at"] <= pd.Timestamp(2020,7,31, tz="Europe/Amsterdam"))]

for sammelstelle, data in first_year.groupby("name"):
    data.sort_values(by=['measured_at'], inplace=True)
    plot_data(data["measured_at"], data["data_distance"], sammelstelle)
    