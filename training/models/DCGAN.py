import torch
from torch import nn

class DCGenerator(nn.Module):
    def __init__(self, latent_dim, image_channels=3):
        super(DCGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),  # Output: [512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),         # Output: [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),         # Output: [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),          # Output: [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),           # Output: [32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, image_channels, 4, 2, 1, bias=False), # Output: [image_channels, 128, 128]
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)



class DCDiscriminator(nn.Module):
    def __init__(self, image_channels=3):
        super(DCDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),       # Output: [64, 64, 64]
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),                   # Output: [128, 32, 32]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),                  # Output: [256, 16, 16]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),                  # Output: [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),                 # Output: [1024, 4, 4]
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),                   # Output: [1, 1, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x).view(-1, 1).squeeze(1)
