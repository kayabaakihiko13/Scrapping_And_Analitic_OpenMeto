import numpy as np
import pandas as pd

from typing import Union
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
        
if __name__ == "__main__":
    series_outliers = detect_outlier(pd.Series([100, 2, 300]))
    print("Outliers in Series:", series_outliers)

