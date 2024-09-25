import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler

class EmgEncoder():
    """SIGNATE NEDO 重心移動予測コンペ用に作成した筋電の前処理クラス
    
    Args:
        num_features (int): 特徴量の数, デフォルトは 16
        num_sequence (int): 時系列の長さ, デフォルトは 1000
        window (int): 移動平均時の窓サイズ, デフォルトは 10
    
    Examples:
        ```
        import scipy.io as sio
        from sklearn.model_selection import train_test_split
        
        # データの読み込み
        ref = sio.loadmat("../data/reference.mat")
        train_x = ref["0005"][0][0][0]
        train_y = ref["0005"][0][0][1]
        train_x, valid_x, train_y, valid_y = train_test_split(
            train_x, train_y, test_size=0.1, random_state=42, shuffle=True
        )
        print(train_x.shape, valid_x.shape)
        >>> (288, 16, 1000) (32, 16, 1000)
        emg_encoder = EmgEncoder()
        train_x = emg_encoder.fit_transform(train_x)
        valid_x = emg_encoder.transform(valid_x)
        print(train_x.shape, valid_x.shape)
        >>> (288, 1000, 16) (32, 1000, 16)
        ```
    
    ## 処理の流れ
        1. データの型を (trials, sequence, features) に整形
        2. 特徴量ごとに移動二乗平均平方根で平滑化
        3. Box-Cox変換を用いて特徴量ごとに対数変換
        4. 特徴量ごとに `MinMaxScaler` で正規化
    
    """
    def __init__(self, num_features=16, num_sequence=1000, window: int=10):
        self.window = window
        self.num_features = num_features
        self.num_sequence = num_sequence
        
    @property
    def __is_fitted__(self):
        try:
            self.params
            return True
        except:
            return False
    
    def fit(self, data: np.ndarray) -> dict:
        # データを (trials, sequence, fetures) に整形
        # data = data.reshape(-1, self.num_sequence, self.num_features)
        data = data.transpose(0, 2, 1)
        
        # 移動二乗平均平方根で平滑化
        data = np.apply_along_axis(self._moveRMS, axis=1, arr=data)
        # Box-Cox変換を適用した際の λ を各特徴量ごとに保存
        lmbda_list = []
        mm_scalers = []
        for i in range(self.num_features):
            xt, lmbda = boxcox(data[:, :, i].flatten())
            mm = MinMaxScaler()
            mm.fit(xt)
            lmbda_list.append(lmbda)
            mm_scalers.append(mm)
        self.params = {"lambda": lmbda_list, "mm_scaler": mm_scalers}
        
        return self.params
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        # データのshapeを整形
        # data = data.reshape(-1, self.num_sequence, self.num_features)
        data = data.transpose(0, 2, 1)
        
        # 移動二乗平均平方根で平滑化
        data = np.apply_along_axis(self._moveRMS, axis=1, arr=data)
        # Box-Cox変換とMinMax正規化を適用
        lmbda_list = []
        mm_scalers = []
        data_transformed = np.zeros_like(data)
        for i in range(self.num_features):
            mm = MinMaxScaler()
            xt, lbmda = boxcox(data[:, :, i].flatten())
            xt = mm.fit_transform(xt.reshape(-1, 1))
            data_transformed[:, :, i] = xt.reshape(-1, self.num_sequence)
            lmbda_list.append(lbmda)
            mm_scalers.append(mm)
        
        self.params = {"lambda": lmbda_list, "mm_scaler": mm_scalers}
        
        return data_transformed
    
    def apply_moveRMS(self, data: np.ndarray) -> np.ndarray:
        data = data.transpose(0, 2, 1)
        # data = data.reshape(-1, self.num_sequence, self.num_features)
        
        # 移動二乗平均平方根で平滑化
        data = np.apply_along_axis(self._moveRMS, axis=1, arr=data)
        
        return data
    
    def transform(self, data: np.ndarray) -> np.ndarray:

        assert self.__is_fitted__, "\033[91m" + "fit() もしくは fit_transform() でスケーラーを事前に定義してください" + "\033[0m"
        
        # moveRMSを適用
        data = data.reshape(-1, self.num_sequence, self.num_features)
        data = np.apply_along_axis(self._moveRMS, axis=1, arr=data)
        
        # box-cox変換とMimMax正規化を適用
        data_transformed = np.zeros_like(data)
        for i in range(self.num_features):
            lmbda = self.params["lambda"][i]
            mm = self.params["mm_scaler"][i]
            xt = boxcox(data[:, :, i], lmbda=lmbda)
            xt = mm.transform(xt.reshape(-1, 1))
            data_transformed[:, :, i] = xt.reshape(-1, self.num_sequence)
            
        return data_transformed
    
    def _filter(self, data, sampling=2000, low=45, high=450):
        return
    
    def _moveRMS(self, data, window=None) -> np.ndarray:
        """データの移動二乗平均平方根を算出

        Args:
            data (array): 入力データ
            window (int, optional): 窓サイズ. Defaults to None.

        Returns:
            np.ndarray: フィルタリング後のデータ
        """
        if window is None:
            window = self.window
        
        return np.convolve(np.sqrt(data*data), np.ones(window)/window, mode="same")

# --------------------
# デバッグ用プログラム
# --------------------
if __name__ == "__main__":
    import scipy.io as sio
    from sklearn.model_selection import train_test_split
    
    # データの読み込み
    ref = sio.loadmat("../data/reference.mat")
    train_x = ref["0005"][0][0][0]
    train_y = ref["0005"][0][0][1]
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.1, random_state=42, shuffle=True
    )
    print(train_x.shape, valid_x.shape)
    # >>> (288, 16, 1000) (32, 16, 1000)
    emg_encoder = EmgEncoder()
    train_x = emg_encoder.fit_transform(train_x)
    valid_x = emg_encoder.transform(valid_x)
    print(train_x.shape, valid_x.shape)
    # >>> (288, 1000, 16) (32, 1000, 16)
