import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init


class Encoder(nn.module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=2 ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=2 ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=2 ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class ColorNet(nn.module):
    def __init__(self, feature_dim, class_num):
        super(ColorNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = out.reshape(x.size(0), -1)
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = nn.Sigmoid(out)
        return out
    
class SegmentationNet(nn.module):
    def __init__(self, feature_dim):
        super(SegmentationNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(16*2 + 3, 16, 3, stride=2),  # b, 16, 5, 5
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.BatchNorm2d(8),
            nn.ReLU(True)
            )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

    
