import numpy as np
import pandas as pd
from typing import Union,Tuple
from collections import Counter
from sklearn.cluster import DBSCAN,KMeans

def dbscan_grid_search(X_data: Union[pd.DataFrame, np.ndarray],
                       eps_space: np.array = np.arange(0.1, 5, 0.1),
                       min_samples_space: np.array = np.arange(1, 50, 1),
                       min_clust: int = 3,
                       max_clust: int = 6)->Tuple:
    """_summary_

    Args:
        X_data (Union[pd.DataFrame, np.ndarray]): _description_
        eps_space (np.array, optional): _description_. Defaults to np.arange(0.1, 5, 0.1).
        min_samples_space (np.array, optional): _description_. Defaults to np.arange(1, 50, 1).
        min_clust (int, optional): _description_. Defaults to 3.
        max_clust (int, optional): _description_. Defaults to 6.

    Returns:
        _type_: _description_
    """
    n_iterations = 0
    dbscan_clusters = []  # List to store the results
    clst_count = []  # List to store cluster counts
    # initial for dbscore
    best_score:float = 0.0
    best_parameter = (0,0)
    for eps_val in eps_space:
        for samples_val in min_samples_space:
            dbscan_grid = DBSCAN(eps=eps_val, min_samples=samples_val)

            # Fit and predict
            clusters = dbscan_grid.fit_predict(X=X_data)

            # Counting the amount of data in each cluster
            cluster_count = Counter(clusters)

            # Saving the number of clusters
            n_clusters = len(np.unique(clusters)) - 1

            # Increasing the iteration tally with each run of the loop
            n_iterations += 1

            # Appending the list each time n_clusters criteria is reached
            if min_clust <= n_clusters <= max_clust:
                dbscan_clusters.append([eps_val, samples_val, n_clusters])
                clst_count.append(cluster_count)
                if n_clusters > best_score:
                    best_score = n_clusters
                    best_parameter = (eps_val,samples_val)
    return dbscan_clusters, clst_count,best_score,best_parameter


def kmean_grid_search(data: pd.DataFrame, k_values: range, init="k-means++", n_init="warn",
                      max_iter=300, tol=0.0001, verbose=0,
                      random_state=None, copy_x=True):
    # inertia value
    inertia_values = []  # Within-cluster sum of squares (inertia)
    best_k = None
    best_inertia = float('inf')

    # Perform the grid search
    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x
        )
        kmeans.fit(data)
        inertia = kmeans.inertia_
        inertia_values.append(inertia)
        
        # Track the best k based on the lowest inertia
        if inertia < best_inertia:
            best_inertia = inertia
            best_k = k
    return best_k
