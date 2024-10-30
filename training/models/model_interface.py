import argparse
import pytorch_lightning as pl
import torch
from models.DC_GAN import DC_Generator, DC_Discriminator

class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        model_name = kwargs["model_name"]
        latent_dim = kwargs["latent_dim"]
        lr = kwargs["lr"]
        
        # 设置生成器和判别器
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
        
        # 生成潜在向量并添加噪声
        z = torch.randn(batch_size, self.latent_dim, 1, 1).type_as(real_images)
        noise = torch.randn_like(z) * 0.1  # 可以调整噪声的标准差
        z += noise  # 添加噪声

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


    def on_epoch_end(self):
        avg_d_loss = torch.stack([x['d_loss'] for x in self.trainer.callback_metrics['train_loss']]).mean()
        avg_g_loss = torch.stack([x['g_loss'] for x in self.trainer.callback_metrics['train_loss']]).mean()
        self.log("avg_d_loss", avg_d_loss)
        self.log("avg_g_loss", avg_g_loss)


    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=30, gamma=0.1)
        scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=30, gamma=0.1)
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

