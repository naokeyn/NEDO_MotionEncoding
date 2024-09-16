import numpy as np
from scipy.stats import boxcox

class PreprocessEMG():
    def __init__(self, window: int=50):
        self.window = window
        
    def fit(self, data):
        pass
    
    def fit_transform(self, data):
        return
    
    def transform(self, data):
        return

    def _filter(data, sampling=2000, low=45, high=450):
        return
    
    def _moveRMS(data, window=25) -> np.ndarray:
        """データの移動二乗平均平方根を算出

        Args:
            data (array): 入力データ
            window (int, optional): 窓サイズ. Defaults to 25.

        Returns:
            np.ndarray: フィルタリング後のデータ
        """
        return np.convolve(np.sqrt(data**2), np.ones(window)/window, mode="same")
