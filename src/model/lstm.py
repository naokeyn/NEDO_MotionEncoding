import torch
from torch import nn

class CNN2LSTM(nn.Module):
    def __init__(self, feature_size, hidden_size, len_sequence, n_layers):
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
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size//4,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_size//4)
        self.relu1 = nn.ReLU()
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size//4,
            hidden_size=hidden_size//4,
            num_layers=n_layers
        )
        
        self.linear = nn.Linear(hidden_size//4, 1)

    def forward(self, x):
        h, *_ = self.lstm1(x)
        # print("lstm output: ", h.size())
        h = self.conv1(h.transpose(2, 1))
        # print("conv output:", h.size())
        h = self.bn1(h)
        h = self.relu1(h)
        h, *_ = self.lstm2(h.transpose(2, 1))
        # print("lstm2 output:", h.size())
        h = self.linear(h[:, -1, :])
        return h


if __name__ == "__main__":
    torch.manual_seed(0)
    dummy_input = torch.randn(1, 1000, 16)      # (batch_size, len_sequence, feature_size)
    params = {
        "feature_size": 16,
        "hidden_size": 32,
        "len_sequence": 1000,
        "n_layers": 1
    }
    print(params)
    model = CNN2LSTM(**params)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.size()}, Output shape: {output.size()}")
