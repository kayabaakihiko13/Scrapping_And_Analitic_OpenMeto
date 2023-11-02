import numpy as np
import pandas as pd
from typing import Union,Tuple
from collections import Counter
from sklearn.cluster import DBSCAN,KMeans
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

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

def custom_train_test_split(data: Union[pd.DataFrame, pd.Series], 
                            train_size: float = None, test_size: float = None) :
    """_summary_
    this function how to spit data to porpuse data
    Args:
       data (Union[pd.DataFrame, pd.Series]) = input data for to split
        trainsize (float) = this for input size for size of train data
        testsize (float) = this for input size for size of test data

    Returns:
        (np.ndarray, np.ndarray): _description_

    ## Example
    >>> data=pd.Series([1,2,3,10,4,5],name="data")
    >>> train,test = custom_train_test_split(data,0.6)
    >>> print(train.shape,test.shape)
    (3, 1) (2, 1)
    >>> print(data.shape)
    (6,)
    """
    
    if train_size is None:
        train_size = 1 - test_size
    elif test_size is None:
        test_size = 1 - train_size
    else:
        raise ValueError("Perhatikan nilai nya")

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("data should be a pandas DataFrame or Series.")
    
    if train_size < 0 or test_size < 0:
        raise ValueError("Train size and test size must be non-negative.")
    
    if train_size + test_size != 1.0:
        raise ValueError("The sum of train_size and test_size must be 1.0.")

    data_length = len(data)
    train_length = int(data_length * train_size)
    test_length = int(data_length * test_size)
    
    if isinstance(data, pd.Series):
        data = data.to_frame()  # Convert Series to DataFrame for slicing
    
    train_data = data.iloc[:train_length].values.reshape(-1,1)
    test_data = data.iloc[train_length:train_length + test_length].values.reshape(-1,1)
    
    return train_data, test_data

class ARITMA:
    def __init__(self, p_value:int, q_value:int, d_value:int):
        self.p_value = p_value
        self.q_value = q_value
        self.d_value = d_value
        self.model = None

    def fit_arima_model(self, data:pd.DataFrame):
        self.model = sm.tsa.ARIMA(data, order=(self.p_value, self.d_value, self.q_value))
        model_results = self.model.fit()
        return model_results

    def forecast_arima_model(self, model_results, steps:int):
        forecast = model_results.forecast(steps=steps)
        forecast_result = forecast.values
        return forecast_result

    def plot_predict(self, model_results, actual_data, steps:int):
        forecast = model_results.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        data_index = model_results.fittedvalues.index
        last_date = data_index[-1]
        forecast_index = pd.date_range(start=last_date, periods=steps, freq=data_index.freq)
        plt.figure(figsize=(10, 6))
        plt.plot(forecast_index, forecast_mean, label='Prediksi', color='blue')
        plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='blue', alpha=0.15, label='Interval Kepercayaan')
        plt.plot(actual_data.index, actual_data.values, label='Actual', color='green')
        plt.legend()
        plt.xlabel('Indeks Waktu')
        plt.ylabel('Nilai Prediksi / Actual')
        plt.title(f'Plot Prediksi Model ARIMA vs Actual feature {actual_data.name}')
        plt.show()

    def accuracy_model(self, test_data):
        if self.model is None:
            raise ValueError("ARIMA model has not been fitted. Please fit the model first.")
        model_results = self.model.fit()
        forecast_mean = self.forecast_arima_model(model_results, steps=len(test_data))
        mae = mean_absolute_error(test_data, forecast_mean)
        mape = np.mean(np.abs(forecast_mean - test_data) / np.abs(test_data))
        result = {"mae": mae, "mape": mape}
        return result

if __name__ =="__main__":
    import pandas as pd
    date_rng = pd.date_range(start='2023-01-01', end='2023-10-28', freq='D')
    pm10_values = np.random.uniform(0, 60, size=len(date_rng))
    data = pd.Series(pm10_values, index=date_rng)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]
    # Fit an ARIMA model to your data
    p_value = 2
    q_value = 2
    d_value = 0
    model =ARITMA(p_value, d_value, q_value)
    model_result = model.fit_arima_model(data)
    model.plot_predict(model_result, test_data, steps=3)
    accuracy = model.accuracy_model(test_data)
    print(len(test_data))
    print("Mean Absolute Error (MAE):", accuracy["mae"])
    print("Mean Absolute Percentage Error (MAPE):", accuracy["mape"])
    
    
