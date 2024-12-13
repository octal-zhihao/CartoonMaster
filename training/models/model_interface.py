import argparse
import pytorch_lightning as pl
import torch
from .Attention_DCGAN import Attention_DCGenerator, Attention_DCDiscriminator
from .DCGAN import Generator as DCGenerator, Discriminator as DCDiscriminator
from .EBGAN import Generator as EBGenerator, Discriminator as EBDiscriminator
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model_name = kwargs["model_name"]
        self.g_lr = kwargs["g_lr"]
        self.d_lr = kwargs["d_lr"]
        self.margin = max(1, kwargs['batch_size'] / 64.0)
        self.latent_dim = kwargs["latent_dim"]

        total_model = {
            'DCGAN': [DCGenerator, DCDiscriminator],
            'Attention_DCGAN': [Attention_DCGenerator, Attention_DCDiscriminator],
            'EBGAN': [EBGenerator, EBDiscriminator]
        }
        if self.model_name not in total_model:
            raise ValueError(f"Model {self.model_name} not supported")
        else:
            self.generator = total_model[self.model_name][0](**kwargs)
            self.discriminator = total_model[self.model_name][1](**kwargs)
 
        self.automatic_optimization = False  # 禁用自动优化

    def forward(self, z):
        return self.generator(z)

    def generator_loss(self, fake_preds):
        return F.binary_cross_entropy(fake_preds, torch.ones_like(fake_preds))

    # 通过减少生成样本之间的相似性提升多样性
    def pullaway_loss(self, embeddings):
        norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
        normalized_emb = embeddings / norm
        similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
        batch_size = embeddings.size(0)
        loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return loss_pt
    
    # EBGAN 生成器损失
    def EB_generator_loss(self, recon_imgs, fake_images, img_embeddings, lambda_pt=0.1):
        mse_loss = nn.MSELoss()
        g_loss = mse_loss(recon_imgs, fake_images.detach()) + lambda_pt * self.pullaway_loss(img_embeddings)
        return g_loss

    def discriminator_loss(self, real_preds, fake_preds):
        true_loss = F.binary_cross_entropy(real_preds, torch.ones_like(real_preds))
        fake_loss = F.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))
        return true_loss + fake_loss

    # EBGAN 判别器损失
    def EB_discriminator_loss(self, real_recon, fake_recon, real_images, fake_images):
        mse_loss = nn.MSELoss()
        d_loss_real = mse_loss(real_recon, real_images)
        d_loss_fake = mse_loss(fake_recon, fake_images)
        d_loss = d_loss_real
        if (self.margin - d_loss_fake.data).item() > 0:
            d_loss += self.margin - d_loss_fake
        return d_loss

    def training_step(self, batch, batch_idx):
        real_images = batch
        batch_size = real_images.size(0)
        # 获取优化器
        opt_g, opt_d = self.optimizers()

        if self.model_name == "EBGAN":
            # ---- Discriminator Step ----
            z = torch.randn(batch_size, self.latent_dim).type_as(real_images)
            fake_images = self(z).detach()  # 生成假图像并阻止生成器更新
            real_recon, _ = self.discriminator(real_images)
            fake_recon, _ = self.discriminator(fake_images)
            d_loss = self.EB_discriminator_loss(real_recon, fake_recon, real_images, fake_images)

            # 更新判别器
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()

            # ---- Generator Step ----
            fake_images = self(z)  # 生成新的假图像用于生成器更新
            recon_imgs, img_embeddings = self.discriminator(fake_images)
            # 计算生成器损失
            g_loss = self.EB_generator_loss(recon_imgs, fake_images, img_embeddings)

            # 更新生成器
            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()

        else:

            # ---- Discriminator Step ----
            z = torch.randn(batch_size, self.latent_dim, 1, 1).type_as(real_images)
            fake_images = self(z).detach()  # 生成假图像并阻止生成器更新
            real_preds = self.discriminator(real_images)
            fake_preds = self.discriminator(fake_images)

            # 计算判别器损失
            d_loss = self.discriminator_loss(real_preds, fake_preds)

            # 更新判别器
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()

            # ---- Generator Step ----
            fake_images = self(z)  # 生成新的假图像用于生成器更新
            fake_preds = self.discriminator(fake_images)  # 通过判别器评估假图像

            # 计算生成器损失
            g_loss = self.generator_loss(fake_preds)

            # 更新生成器
            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()

        self.log("d_loss", d_loss, prog_bar=True)
        self.log("g_loss", g_loss, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        real_images = batch
        batch_size = real_images.size(0)
        
        if self.model_name == "EBGAN":
            z = torch.randn(batch_size, self.latent_dim).type_as(real_images)
            # 计算判别器的预测并计算损失
            real_recon, _ = self.discriminator(real_images)
            fake_images = self(z).detach()
            fake_recon, _ = self.discriminator(fake_images)
            val_d_loss = self.EB_discriminator_loss(real_recon, fake_recon, real_images, fake_images)
            # 计算生成器的输出和判别器的预测
            fake_images = self(z)
            recon_imgs, img_embeddings = self.discriminator(fake_images)
            # 计算生成器损失
            val_g_loss = self.EB_generator_loss(recon_imgs, fake_images, img_embeddings)
        else:
            z = torch.randn(batch_size, self.latent_dim, 1, 1).type_as(real_images)
            # 计算生成器的输出和判别器的预测
            fake_images = self(z)
            fake_preds = self.discriminator(fake_images)
            # 计算生成器损失
            val_g_loss = self.generator_loss(fake_preds)
            # 计算判别器的预测并计算损失
            real_preds = self.discriminator(real_images)
            val_d_loss = self.discriminator_loss(real_preds, fake_preds)
        self.log("val_g_loss", val_g_loss, prog_bar=True)
        self.log("val_d_loss", val_d_loss, prog_bar=True)

    def on_epoch_end(self):
        avg_d_loss = torch.stack([x['d_loss'] for x in self.trainer.callback_metrics['d_loss']]).mean()
        avg_g_loss = torch.stack([x['g_loss'] for x in self.trainer.callback_metrics['g_loss']]).mean()
        self.log("avg_d_loss", avg_d_loss, prog_bar=True)
        self.log("avg_g_loss", avg_g_loss, prog_bar=True)


    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=self.d_lr, betas=(0.5, 0.999))
        return opt_g, opt_d

