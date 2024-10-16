import torch
from torch import nn

class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        channel = self.in_channels // 4
        
        self.conv1  = nn.Conv1d(self.in_channels, channel, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(channel)
        self.relu1  = nn.ReLU()
        self.conv2  = nn.Conv1d(channel, channel, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm1d(channel)
        self.relu2  = nn.ReLU()
        self.conv3  = nn.Conv1d(channel, self.out_channels, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(self.out_channels)
        self.relu3  = nn.ReLU()
    
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        
        if self.in_channels != self.out_channels:
            return self.relu3(h)
        
        return self.relu3(h + x)

class ResNet1d(nn.Module):
    def __init__(self, in_channels=16, first_conv_out_channels=32, out_channels=1, num_layers=[3, 4, 6, 3]):
        super().__init__()
        self.in_channels = in_channels
        self.first_conv_out_channels = first_conv_out_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.conv1 = nn.Conv1d(
            self.in_channels, 
            self.first_conv_out_channels, 
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.bn1 = nn.BatchNorm1d(self.first_conv_out_channels)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block1 = self._build_block(num_layers=self.num_layers[0], in_channels=self.first_conv_out_channels, out_channels=self.first_conv_out_channels*2)
        self.block2 = self._build_block(num_layers=self.num_layers[1], in_channels=self.first_conv_out_channels*2, out_channels=self.first_conv_out_channels*4)
        self.block3 = self._build_block(num_layers=self.num_layers[2], in_channels=self.first_conv_out_channels*4, out_channels=self.first_conv_out_channels*8)
        self.block4 = self._build_block(num_layers=self.num_layers[3], in_channels=self.first_conv_out_channels*8, out_channels=self.first_conv_out_channels*16)
        self.average_pool = nn.AdaptiveAvgPool1d(output_size=self.out_channels)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.first_conv_out_channels*16, 30)
        
    def forward(self, x):
        
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.maxpool(h)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.average_pool(h)
        h = self.flatten(h)
        h = self.linear(h)
        
        return h
    
    def _build_block(self, num_layers, in_channels, out_channels):
        block = nn.Sequential()
        for _ in range(num_layers-1):
            block.append(ResBlock1d(in_channels, in_channels))
        block.append(ResBlock1d(in_channels, out_channels))
        return block
    
def test():
    model = ResNet1d()
    x = torch.randn(30, 16, 1000)
    y = model(x)
    print("Input Shape:", x.size())
    print("Output Shape:", y.size())

if __name__ == "__main__":
    test()
