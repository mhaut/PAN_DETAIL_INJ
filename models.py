import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class DiCNN1(nn.Module):
    def __init__(self, input_channels):
        super(DiCNN1, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        
    def forward(self, input_ms_pan):
        input_ms, input_pan = torch.split(input_ms_pan, [input_ms_pan.shape[1] - 1, 1], 1)
        x = F.relu(self.conv1(input_ms_pan))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return (x[:,:4,:,:] + input_ms)



class DiCNN2(nn.Module):
    def __init__(self, input_channels):
        super(DiCNN2, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)

        
    def forward(self, input_ms_pan):
        input_ms, input_pan = torch.split(input_ms_pan, [input_ms_pan.shape[1] - 1, 1], 1)
        x = F.relu(self.conv1(input_ms))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return input_ms + x
