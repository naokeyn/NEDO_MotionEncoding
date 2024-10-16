import os, json, gc
import scipy.io as sio
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

NUM_EMG_CHANNELS: int = 16
NUM_AXIS: int = 3
INPUT_LENGHT: int = 1000
OUTPUT_LENGHT: int = 30
USERS: list = ["0001", "0002", "0003", "0004"]

def _load_dataset(datadir="/app/data"):
    train = sio.loadmat(os.path.join(datadir, "train.mat"))
    test = sio.loadmat(os.path.join(datadir, "test.mat"))
    
    return train, test
    
def pred_1point_data(return_scalers=True):
    train, test = _load_dataset()
    X, Y, test_X = [], [], []
    
    # ユーザーごとに前処理を行う
    input_scalers = {}
    output_scalers = {}
    for user in USERS:
        train_x = train[user][0][0][0]
        train_y = train[user][0][0][1]
        test_x = test[user][0][0][0]
        
        # EMGのチャンネルごとに正規化
        input_scaler = {}
        for channel_id in range(NUM_EMG_CHANNELS):
            _max = np.max(train_x[:, channel_id, :])
            _min = np.min(train_x[:, channel_id, :])
            train_x[:, channel_id, :] = (train_x[:, channel_id, :] - _min) / (_max - _min)
            test_x[:, channel_id, :] = (test_x[:, channel_id, :] - _min) / (_max - _min)
            input_scaler[channel_id] = (_min, _max)
        input_scalers[user] = input_scaler
        
        # 加速度の軸ごとに正規化
        output_scaler = {}
        for axis in range(NUM_AXIS):
            _max = np.max(train_y[:, axis, :])
            _min = np.min(train_y[:, axis, :])
            train_y[:, axis, :] = (train_y[:, axis, :] - _min) / (_max - _min)
            output_scaler[axis] = (_min, _max)
        output_scalers[user] = output_scaler
        
        # 入出力を分割して保存
        trial_x = []
        trial_y = []
        trial_test_x = []
        for _x, _y, _x_test in zip(train_x, train_y, test_x):
            _x = np.array_split(_x[:, :990], 30, axis=1)
            _y = np.split(_y, 30, axis=1)
            _x_test = np.array_split(_x_test[:, :990], 30, axis=1)
            
            trial_x += _x
            trial_y += _y
            trial_test_x += _x_test
        X += trial_x
        Y += trial_y
        test_X += trial_test_x


    X = np.array(X)
    Y = np.array(Y)
    test_X = np.array(test_X)

    if return_scalers:
        return X, Y, test_X, input_scalers, output_scalers
    return X, Y, test_X


def pred_30points_data(return_scalers=True):
    X, Y, test_X = [], [], []
    train, test = _load_dataset()
    
    
    if return_scalers:
        return X, Y, test_X, input_scalers, output_scalers
    return X, Y, test_X

if __name__ == "__main__":
    X, y, test_x, input_scaler, output_scaler = pred_1point_data()
    print(X.shape, y.shape, test_x.shape)
    dummy_pred = np.random.random(size=(10, 30, 1))     # (len_data, len_sequence, axis)
    
