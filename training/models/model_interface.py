import argparse
import pytorch_lightning as pl
import torch
from models.DC_GAN import DC_Generator, DC_Discriminator
from data.data_interface import DInterface

class MInterface(pl.LightningModule):
    def __init__(self, latent_dim, lr, model_name="DC_GAN"):
        super().__init__()
        if model_name == "DC_GAN":
            self.generator = DC_Generator(latent_dim)
            self.discriminator = DC_Discriminator()
        else:
            raise ValueError(f"Model {model_name} not supported")
        self.latent_dim = latent_dim
        self.lr = lr
        self.automatic_optimization = False  # 禁用自动优化

    def forward(self, z):
        return self.generator(z)

    def generator_loss(self, fake_preds):
        return torch.mean((fake_preds - 1) ** 2)

    def discriminator_loss(self, real_preds, fake_preds):
        real_loss = torch.mean((real_preds - 1) ** 2)
        fake_loss = torch.mean(fake_preds ** 2)
        return (real_loss + fake_loss) / 2

    def training_step(self, batch, batch_idx):
        real_images = batch
        batch_size = real_images.size(0)
        z = torch.randn(batch_size, self.latent_dim, 1, 1).type_as(real_images)

        # 获取优化器
        opt_g, opt_d = self.optimizers()

        # ---- Discriminator Step ----
        fake_images = self(z).detach()  # 阻止生成器更新
        real_preds = self.discriminator(real_images)
        fake_preds = self.discriminator(fake_images)
        d_loss = self.discriminator_loss(real_preds, fake_preds)

        # 更新判别器
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # ---- Generator Step ----
        fake_images = self(z)  # 生成新图像用于生成器更新
        fake_preds = self.discriminator(fake_images)
        g_loss = self.generator_loss(fake_preds)

        # 更新生成器
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("d_loss", d_loss, prog_bar=True)
        self.log("g_loss", g_loss, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN for cartoon image generation")
    parser.add_argument("--data_path", type=str, default="./dataset/cartoon_color", help="Path to cartoon image dataset")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()

    # 使用 DInterface 加载数据
    data_module = DInterface(data_path=args.data_path, batch_size=args.batch_size)

    # 初始化模型
    model = GANTrainer(latent_dim=args.latent_dim, lr=args.lr)

    # 训练
    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(model, data_module)
