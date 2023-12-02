import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd
import geoplot
import json
import re
import sys
from shapely.geometry import shape, Point, GeometryCollection, MultiPoint, Polygon
from shapely.ops import voronoi_diagram
from pathlib import Path
from csv import DictWriter

bevölkerungsdaten_file = "C:/Users/cedri/OneDrive - OST/Hackathon 2023/Bevölkerungsdaten nach Quartier.csv"
wohnviertel_file = "C:/Users/cedri/OneDrive - OST/Hackathon 2023/wohnviertel.geojson"
sammelstellen_file = "C:/Users/cedri/OneDrive - OST/Hackathon 2023/fuellstandsensoren-glassammelstellen-weissglas.csv"


quartier_counts = pd.read_csv(bevölkerungsdaten_file, sep=";")
quartier_counts = quartier_counts[["GEBIET_NAME", "INDIKATOR_JAHR", "INDIKATOR_VALUE"]]

quartier_counts = quartier_counts[quartier_counts["GEBIET_NAME"].str.contains("Quartier ")]

quartier_counts['GEBIET_NAME'] = quartier_counts['GEBIET_NAME'].str.replace("Quartier ","")

with open(wohnviertel_file) as f:
  features = json.load(f)["features"]

# NOTE: buffer(0) is a trick for fixing scenarios where polygons have overlapping coordinates 

location_dict = {}
location_list = []
for feature in features:
    properties = feature["properties"]
    polygon = Polygon(feature["geometry"]["coordinates"][0])
    if properties["statistisc"] == "Linsenbühl-Dreilinden":
        location_dict["Linsebühl-Dreilinden"] = {"geometry": polygon, "population": {}}
    elif properties["statistisc"] == "Rosenberg-Kreubleiche":
        location_dict["Rosenberg-Kreuzbleiche"] = {"geometry": polygon, "population": {}}
    else:
        location_dict[properties["statistisc"]] = {"geometry": polygon, "population": {}}

for index, data in quartier_counts.iterrows():
    location_dict[data["GEBIET_NAME"]]["population"][data["INDIKATOR_JAHR"]] = data["INDIKATOR_VALUE"]


sensor_data = pd.read_csv(sammelstellen_file, delimiter=";")


geodata = {}
geodata_list = []
all_points = []
collection_points = sensor_data.drop_duplicates(subset=["name"])
for _, collection_point in collection_points.iterrows():
    regex_result = re.search(r"\[([\d\.]+), ([\d\.]+)\]",collection_point["location"])
    point_list = [regex_result.group(1),regex_result.group(2)]
    point = Point(point_list)
    all_points.append(point)
    geodata_list.append({"name":collection_point["name"], "geom":point})

multipoint = MultiPoint(all_points)
voronoi = voronoi_diagram(multipoint)

output = Path("testing_output")
counter = 0
for polygon in voronoi.geoms:
    for index, point in enumerate(geodata_list):
        if polygon.contains(point["geom"]):
            geodata_list[index]["polygon"] = polygon
    counter += 1

result = []
for collection_point in geodata_list:
    calculation_dict = {"name":collection_point["name"], "years":{}}
    gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[collection_point["polygon"]])
    gdf.to_file(output / "sammelstelle.shp")
    for population in location_dict.values():
        if collection_point["polygon"].intersects(population["geometry"]):
            factor = ((population["geometry"].intersection(collection_point["polygon"]).area/population["geometry"].area))
            for year in population["population"].keys():
                if not calculation_dict.get(year):
                    calculation_dict["years"][year] = 0
                calculation_dict["years"][year] += population["population"][year] * factor
    result.append(calculation_dict)

with open("population.csv", "w", newline="") as w:
    dict_writer = DictWriter(w, ["name","year","population"])
    dict_writer.writeheader()
    for row in result:
        for year in row["years"]:
            insert = {"name": row["name"], "year": year, "population":row["years"][year]}
            dict_writer.writerow(insert)