from Tengai import (dataset,model,preprocessing,visual_series)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt

def city_long_lat(city_name: str):
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

class DemoCluster:
    
    def __init__(self, data, json_file):
        self.data = data
        self.label_cluster = json_file
    
    def load_cluster_labels(self):
        with open(self.label_cluster, 'r') as f:
            cluster_labels = json.load(f)
        return cluster_labels
    
    def visual_cluster(self):
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
        cluster_labels = self.load_cluster_labels()
        cluster_labels = {int(k): v for k, v in cluster_labels.items()}  # Convert keys to integers

        # Create a bar plot with custom cluster labels
        cluster_labels_list = [cluster_labels[cluster] for cluster in unique_clusters]
        plt.bar(cluster_labels_list, cluster_counts)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Data Points')
        plt.xticks(rotation=45)  # Rotate X-axis labels for better readability
        plt.title('Group Cluster Weather in City')
        plt.show()

class TimeSeries_Demo:
    def __init__(self, data,feature):
        self.time_series = preprocessing.clustering_DataFrame(data, "time")[feature]
        
    def visual_time_series(self, steps: int = 3):
        train_size = int(len(self.time_series) * 0.7)
        test_data = self.time_series.iloc[train_size:]
        model_aritma = model.ARITMA(1, 1, 0)
        model_results = model_aritma.fit_arima_model(self.time_series)
        model_aritma.plot_predict(model_results, test_data, steps)

if __name__ == "__main__":
    with open("jupyter\cluster_mode.pkl", "rb") as f:
        cluster_assignments = joblib.load(f)
    lat,long = city_long_lat("jakarta")
    data = dataset.AirQuality(lat,long)
    demo_time_series = TimeSeries_Demo(data,"pm10")
    demo_time_series.visual_time_series(5)
    