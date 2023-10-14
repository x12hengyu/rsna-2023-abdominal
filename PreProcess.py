import torch
from torch.nn import Module
from torch import nn
import pdb

class CompleteFilter(Module):
    """

    """
    def __init__(self, max_channel: int):
        super().__init__()
        # fix [C, H, W]

        # [512, 512, max_channel] -> [256, 256, 32]
        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels = max_channel, out_channels = 32, kernel_size = 5, stride = 2, padding = 2),
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                nn.MaxPool2d(kernel_size = 2, stride = 2)
                )

        # [256, 256, 32] -> [16, 16, 128]
        self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 2, padding = 2),
                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 2, padding = 2),
                nn.MaxPool2d(kernel_size = 2, stride = 2)
                )

        self.head = nn.Sequential(
                nn.Flatten(start_dim = 1, end_dim = -1),
                nn.LazyLinear(out_features = 4096),
                nn.LazyLinear(out_features = 1024),
                nn.LazyLinear(out_features = 1))

    def forward(self, x):
        # pdb.set_trace()
        x = self.block_1(x)
        x = self.block_2(x)
        return self.head(x).squeeze()

# model = CompleteFilter(1727)
# model = model.to(torch.device('cuda:7'))
# input = torch.randn(8, 1727, 512, 512).to(torch.device('cuda:7'))
# out = model(input)
# print(out.shape)