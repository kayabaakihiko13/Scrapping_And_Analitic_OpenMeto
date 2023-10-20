import numpy as np
import pandas as pd
from typing import Union,Tuple
from collections import Counter
from sklearn.cluster import DBSCAN,KMeans
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests

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

# Model Time Series
def grangers_causation_matrix(data: pd.DataFrame, variables: list, maxlag=7, test='ssr_chi2test', verbose=False):    
    """
    ## Describes

    Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    ## Parameters
    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables)), dtype=float), columns=variables, index=variables)
    
    for c in df.columns:
        for r in df.index:
            if c != r:
                test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
                if verbose:
                    print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
    
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
