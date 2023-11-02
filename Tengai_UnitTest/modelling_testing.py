import unittest
from unittest.mock import patch
from Tengai.model import ARITMA
from Tengai.preprocessing import Clustering_Optimalization
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class TestingModeling(unittest.TestCase):
    
    def ARIMA_model(self):
        date_rng = pd.date_range(start="2002-10-01",end="2002-11-01")
        random_values = np.random.uniform(0,60,size=len(date_rng))
        data = pd.Series(random_values,index=date_rng)
        model = ARITMA(1,1,0)
        model_result = model.fit_arima_model(data)
        forcesting_result = model.forecast_arima_model(model_result,4)
        
        self.assertTrue(len(forcesting_result)==4)
    

class TestClusteringOptimalization(unittest.TestCase):
    def test_elbow_method_plot(self):
        # Buat data acak untuk pengujian
        data = pd.DataFrame(np.random.rand(100, 2), columns=["Feature1", "Feature2"])
        # Sesuaikan dengan rentang yang ingin diuji
        range_iterable = range(1, 11)  
        # Simpan gambar hasil pengujian
        plt.savefig('elbow_plot.jpg')
    
if __name__ =="__main__":
    unittest.main()
        
        