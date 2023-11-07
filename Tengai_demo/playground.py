from Tengai import (dataset,model,preprocessing)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def _city_long_lat(city_name: str):
    city_info = {
        "jakarta": {"lat": -6.1818, "long": 106.8223},
        "bandung": {"lat": -6.9222, "long": 107.6069},
        "surabaya": {"lat": -7.2492, "long": 112.7508},
    }

    # Convert the city name to lowercase to ensure case-insensitive matching
    city_name = city_name.lower()

    if city_name in city_info:
        result = [v for v in city_info[city_name].values()]
        return result
    else:
        raise ValueError("please put city name be correctly")

def load_data_AirQuality(name_city:str) -> pd.DataFrame:
    # to get data lat and long
    lat,long = _city_long_lat(name_city)

    return dataset.AirQuality(lat,long)

class DemoCluster:
    
    def __init__(self, data:np.ndarray, json_file:str=None) -> None:
        self.data = data
        
        self.json_file = "jupyter/label_cluster.json" if json_file is None else json_file

    def __load_cluster_labels(self):
        with open(self.json_file, 'r') as f:
            cluster_labels = json.load(f)
        return cluster_labels
    
    def visual_cluster(self) -> plt.Figure:
        # Define preprocessing and outlier detection functions or import them from a module 
        # Perform clustering
        data_cluster = preprocessing.clustering_DataFrame(self.data, "time")
        no_outliers_count = 0

        for feat in data_cluster.columns:
            outlier = preprocessing.detect_outlier_zscore(data_cluster[feat])
            if any(outlier[0]):
                print(f"Outlier of {feat} is in index: {outlier[0]}")
                data_cluster[feat] = preprocessing.detect_outlier_zscore(data_cluster[feat], change_outlier=True)
            else:
                no_outliers_count += 1
        
        # Scale the data
        scaler = StandardScaler()
        scaler_data = scaler.fit_transform(data_cluster)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=6, random_state=42, init="random", max_iter=100)
        cluster_assignments = kmeans.fit_predict(scaler_data)
        unique_clusters, cluster_counts = np.unique(cluster_assignments, return_counts=True)

        # Load cluster labels from JSON
        cluster_labels = self.__load_cluster_labels()
        cluster_labels = {int(k): v for k, v in cluster_labels.items()}  # Convert keys to integers

        # Create a bar plot with custom cluster labels
        cluster_labels_list = [cluster_labels[cluster] for cluster in unique_clusters]
        fig = px.bar(x=cluster_labels_list,y=cluster_counts,labels={'x': 'Cluster', 
                                                                    'y': 'Number of Data Points'})
        
        fig.update_xaxes(tickangle=45)
        fig.update_layout(
                          title=dict(text="Group Cluster Weather in City", x=0.5),
                          xaxis_title=dict(text='X-Axis Label', standoff=15),  
                          yaxis_title=dict(text='Y-Axis Label', standoff=15), ) 

        fig.show()
class TimeSeries_Demo:
    def __init__(self, data:pd.DataFrame,feature:str):
        self.time_series = preprocessing.clustering_DataFrame(data, "time")[feature]
        
    def visual_time_series(self, steps: int = 3):
        train_size = int(len(self.time_series) * 0.7)
        test_data = self.time_series.iloc[train_size:]
        model_aritma = model.ARITMA(1, 1, 0)
        model_results = model_aritma.fit_arima_model(self.time_series)
        model_aritma.plot_predict(model_results, test_data, steps)

if __name__ == "__main__":
    data = load_data_AirQuality("Jakarta")
    demo = DemoCluster(data)
    demo.visual_cluster()
    