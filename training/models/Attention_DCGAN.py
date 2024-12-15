import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # [B, N, C//8]
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # [B, C//8, N]
        attention = torch.bmm(proj_query, proj_key)  # [B, N, N]
        attention = torch.softmax(attention, dim=-1)
        
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x  # Add residual connection
        return out
    
class Attention_DCGenerator(nn.Module):
    def __init__(self, **kwargs):
        super(Attention_DCGenerator, self).__init__()
        image_channels = kwargs["image_channels"]
        latent_dim = kwargs["latent_dim"]
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

            SelfAttention(64),  # 添加自注意力层
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),           # Output: [32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, image_channels, 4, 2, 1, bias=False), # Output: [image_channels, 128, 128]
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)



class Attention_DCDiscriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Attention_DCDiscriminator, self).__init__()
        image_channels = kwargs["image_channels"]
        self.fc = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),       # Output: [64, 64, 64]
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),                   # Output: [128, 32, 32]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            SelfAttention(128),  # 在判别器中加入自注意力层

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
