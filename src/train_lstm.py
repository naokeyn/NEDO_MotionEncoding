import os, json, gc
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

from tqdm import tqdm

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model import CNN2LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_emg_channels = 16
num_axis = 3
input_length = 1000
output_length = 30
split_width = input_length // output_length

feature_size = 16
len_sequence = 33
hidden_size = 64
n_layers = 1

batch_size = 32
num_workers = 0
world_size = 2
epochs = 50
learning_rate = 10e-2

def prepare_data():
    train = sio.loadmat("../data/train.mat")
    test = sio.loadmat("../data/test.mat")

    X = []
    y = []
    test_X = []

    # ユーザーごとに前処理を行う
    users = ["0001", "0002", "0003", "0004"]
    input_scalers = {}
    output_scalers = {}
    for user in users:
        train_x = train[user][0][0][0]
        train_y = train[user][0][0][1]
        test_x = test[user][0][0][0]
        
        # EMGのチャンネルごとに正規化
        input_scaler = {}
        for channel_id in range(num_emg_channels):
            _max = np.max(train_x[:, channel_id, :])
            _min = np.min(train_x[:, channel_id, :])
            train_x[:, channel_id, :] = (train_x[:, channel_id, :] - _min) / (_max - _min)
            test_x[:, channel_id, :] = (test_x[:, channel_id, :] - _min) / (_max - _min)
            input_scaler[channel_id] = (_min, _max)
        input_scalers[user] = input_scaler
        
        # 加速度の軸ごとに正規化
        output_scaler = {}
        for axis in range(num_axis):
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
        y += trial_y
        test_X += trial_test_x

    X = np.array(X)
    y = np.array(y)
    test_X = np.array(test_X)

    return X, y, test_X

class CustomDataset(Dataset):
    def __init__(self, x, y=None, v_axis: int=0):
        self.x = x
        self.y = y
        self.v_axis = v_axis
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.y is None:
            return torch.tensor(self.x[idx].T, dtype=torch.float32)
        
        return torch.tensor(self.x[idx].T, dtype=torch.float32), torch.tensor(self.y[idx, self.v_axis, :], dtype=torch.float32)

class CustomLSTM(nn.Module):
    def __init__(
        self,
        feature_size: int=16,
        len_sequence: int=33,
        hidden_size: int=32, 
        n_layers: int=1
    ) -> None:
        
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.len_sequence = len_sequence
        self.n_layers = n_layers
        
        self.lstm1 = nn.LSTM(
            input_size=feature_size, 
            hidden_size=hidden_size, 
            num_layers=n_layers,
            batch_first=True
        )
        self.linear1 = nn.Linear(hidden_size, hidden_size//2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size//4, 1)
        
        
    def forward(self, x) -> torch.Tensor:
        output, (hn, cn) = self.lstm1(x)
        
        # 最後の出力のみを用いる 出力のshape: (batch_size, len_sequence, num_features)
        output = self.linear1(output[:, -1, :])
        output = self.relu1(output)
        output = self.linear2(output)
        output = self.relu2(output)
        output = self.linear3(output)
        return output


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
def train(rank, world_size, model, dataloader, learning_rate=10e-3, epochs=30):
    
    def train_1epoch(train_loader):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(rank)
            y = y.to(rank)
            pred = ddp_model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def valid_1epoch(valid_loader):
        total_loss = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(rank)
                y = y.to(rank)
                pred = ddp_model(x)
                total_loss += loss_fn(pred, y).item()
        return total_loss / len(valid_loader)
    
    setup(rank, world_size)
    
    torch.manual_seed(42)
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate)
    train_loader, valid_loader = dataloader
    
    for epoch in tqdm(range(epochs)):
        ddp_model.train()
        train_loss = train_1epoch(train_loader)
        ddp_model.eval()
        valid_loss = valid_1epoch(valid_loader)
        # print(f"[Epoch {epoch}] train-loss: {train_loss:.5f}, valid-loss: {valid_loss:.5f}")
        if rank == 0:
            tqdm.write(f"[Epoch {epoch}] train-loss: {train_loss:.5f}, valid-loss: {valid_loss:.5f}")
    cleanup()

def run(rank, world_size):
    print(f"Rank {rank}/{world_size} is running")
    X, y, test_X = prepare_data()
    dataset = CustomDataset(X, y)
    train_idx, valid_idx = train_test_split(range(len(dataset)), test_size=0.2, shuffle=True, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    valid_sampler = DistributedSampler(
        valid_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=valid_sampler
    )
    # model = CustomLSTM(
    #     feature_size=feature_size,
    #     len_sequence=len_sequence,
    #     hidden_size=hidden_size,
    #     n_layers=n_layers
    # )
    params = {
        "feature_size": 16,
        "hidden_size": 32,
        "len_sequence": 1000,
        "n_layers": 1
    }
    model = CNN2LSTM(**params)
    
    train(rank, world_size, model, (train_loader, valid_loader), learning_rate, epochs)
    
    torch.save(model, "../models/cnn2lstm_test01.pt")
    
if __name__ == "__main__":
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
