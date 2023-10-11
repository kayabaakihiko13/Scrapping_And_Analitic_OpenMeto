import numpy as np
import pandas as pd
from typing import Any,Union
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans,HDBSCAN
import matplotlib.pyplot as plt


def detect_outlier(data:Union[pd.Series,np.array],
                   see_value:bool=None) -> np.array:
    see_value = False if see_value is None else see_value

    """
    ## Description

    this function detection outlier

    Args:
        data (Union[pd.Series,np.array]): input data a from Series or np.array

    Returns:
        np.array: result of index if see_value is True the element
                  is value
    """

    mean_val = data.mean()
    zscore = (data - mean_val) / data.std()
    if isinstance(data,pd.Series):
        if see_value:
            return data[abs(zscore)>3]
        else:
            return data.index[abs(zscore)>3]
    if isinstance(data,np.ndarray):
        if see_value:
            return data[abs(zscore)>3]
        else:
            return np.where(abs(zscore)>3)[0].tolist()
        
def undersampling_data(data:pd.DataFrame,ratio:float=.5)-> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        ratio (float, optional): _description_. Defaults to .5.

    Returns:
        pd.DataFrame: _description_
    """
    n_samples = int(len(data)*ratio)
    undersampled_time_series = data.sample(n=n_samples,random_state=42)
    undersampled_time_series = undersampled_time_series.sort_index()
    return undersampled_time_series

def clustering_DataFrame(data:pd.DataFrame,name_time_feature:str,
                         rule:str="D")->pd.DataFrame:
    """_summary_
    this function about how to result dataframe ready to clustering
    Args:
        data (pd.DataFrame): _description_
        name_time_feature (str): _description_
        rule (str, optional): _description_. Defaults to "D".

    Returns:
        pd.DataFrame: _description_
    """
    # setting time features
    data[name_time_feature]=pd.to_datetime(data[name_time_feature])
    data.set_index(name_time_feature,inplace=True)
    
    return data.resample(rule).mean()


def clustering_data(data: pd.DataFrame,
                    scaling_method: str = "Standard",
                    clustering_method: str = "KMeans",
                    random_state: int = 42,
                    show_optimal: bool = True) -> Any:

    if scaling_method.lower() == "standard":
        scaling_model = StandardScaler()

    if clustering_method.lower() == "kmeans":
        clustering_model = KMeans(random_state=random_state)
    # Add more clustering methods as needed

    # Apply scaling and clustering in a pipeline
    pipeline = make_pipeline(scaling_model, clustering_model)
    labels = pipeline.fit_predict(data)

    if show_optimal and clustering_method.lower() == "kmeans":
        inertia = []
        # Try different values of k from 1 to 10
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)

        # Identify the elbow point
        diff_inertia = [inertia[i] - inertia[i - 1] for i in range(1, len(inertia))]
        elbow_index = diff_inertia.index(max(diff_inertia)) + 1

        # Plot the elbow graph
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.legend()
        plt.show()


