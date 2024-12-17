from torch import nn


class Generator(nn.Module):
    def __init__(self, **cfg):
        super(Generator, self).__init__()
        self.img_size = cfg['image_size']
        self.latent_dim = cfg['latent_dim']
        self.channels = cfg['image_channels']
        self.init_size = self.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, **cfg):
        super(Discriminator, self).__init__()
        self.img_size = cfg['image_size']
        self.channels = cfg['image_channels']
        self.down = nn.Sequential(nn.Conv2d(self.channels, 64, 3, 2, 1), nn.ReLU())
        self.down_size = self.img_size // 2
        down_dim = 64 * (self.img_size // 2) ** 2

        self.embedding = nn.Linear(down_dim, 32)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, self.channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        embedding = self.embedding(out.view(out.size(0), -1))
        out = self.fc(embedding)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out, embedding