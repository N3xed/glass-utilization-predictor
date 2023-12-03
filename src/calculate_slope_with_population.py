import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt



basepath = Path("C:/Users/cedri/OneDrive - OST/Hackathon 2023/")

whiteglass_population_path = basepath / 'population_weisglas.csv'
whiteglass_slopes_path = basepath / 'weisglas_slopes.csv'

braunglas_population_path = basepath / 'population_braunglas.csv'
braunglas_slopes_path = basepath / 'braunglas_slopes.csv'

greenglass_population_path = basepath / 'population_grunglas.csv'
greenglass_slopes_path = basepath / 'grunglas_slopes.csv'

for glass_type in ["weisglas","braunglas","grunglas"]:
    if glass_type == "weisglas":
        population_path = whiteglass_population_path
        slope_path = whiteglass_slopes_path
    elif glass_type == "braunglas":
        population_path = braunglas_population_path
        slope_path = braunglas_slopes_path
    elif glass_type == "grunglas":
        population_path = greenglass_population_path
        slope_path = greenglass_slopes_path
    else:
        raise NotImplementedError()
    data_population = pd.read_csv(population_path, delimiter=",")
    data_slopes = pd.read_csv(slope_path, delimiter=",")
    
    temp_dict = {}
    for index, data in data_slopes.iterrows():
        temp_dict[data["name"]] = {"slope":data["slope (fullness/day)"]}
        
    
    for index, data in data_population.iterrows():
        if data["name"] in temp_dict:
            temp_dict[data["name"]].update({"population":data["population"]})
    print(temp_dict)
    
    x = []
    y = []
    for value in temp_dict.values():
        x.append(value["slope"])
        y.append(value["population"])
    
    np_x = np.array(x)
    np_y = np.array(y)
    # plt.scatter(np_x, np_y)
    # plt.show()
    
    correlation_coefficient = np.corrcoef(np_x, np_y)[0, 1]

    plt.scatter(np_x, np_y, label=f'Correlation: {correlation_coefficient:.2f}')
    plt.legend()
    plt.title(f"{glass_type}")
    plt.show()
    
    