import argparse
import pytorch_lightning as pl
import torch
from .Attention_DCGAN import Attention_DCGenerator, Attention_DCDiscriminator
from .DCGAN import DCGenerator, DCDiscriminator
import torch.nn.functional as F
import copy


class UnrolledMInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        model_name = kwargs["model_name"]
        self.latent_dim = kwargs["latent_dim"]
        self.g_lr = kwargs["g_lr"]
        self.d_lr = kwargs["d_lr"]

        # 设置生成器和判别器
        self.generator = DCGenerator(self.latent_dim)
        self.discriminator = DCDiscriminator()
        self.automatic_optimization = False  # 禁用自动优化

        self.unrolled_step = 3
        self.cnt = 0
        self.backup = None

    def forward(self, z):
        return self.generator(z)

    def generator_loss(self, fake_preds):
        # 生成器的目标是让判别器将生成的图像判别为真实
        return F.binary_cross_entropy(fake_preds, torch.ones_like(fake_preds))

    def discriminator_loss(self, real_preds, fake_preds):
        # 判别器损失由真实样本的损失和假样本的损失组成
        true_loss = F.binary_cross_entropy(real_preds, torch.ones_like(real_preds))
        fake_loss = F.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))
        return true_loss + fake_loss

    def training_step(self, batch, batch_idx):
        real_images = batch
        batch_size = real_images.size(0)

        # 生成潜在向量
        z = torch.randn(batch_size, self.latent_dim, 1, 1).type_as(real_images)

        # 获取优化器
        opt_g, opt_d = self.optimizers()

        # ---- Discriminator Step ----
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

        # 判别器更新1次权重，生成器更新3次权重
        if self.cnt == 0:
            self.backup = copy.deepcopy(self.discriminator.state_dict())
        elif self.cnt == self.unrolled_step - 1:
            self.discriminator.load_state_dict(self.backup)
        self.cnt = (self.cnt + 1) % self.unrolled_step

    def validation_step(self, batch, batch_idx):
        real_images = batch
        batch_size = real_images.size(0)

        # 生成潜在向量
        z = torch.randn(batch_size, self.latent_dim, 1, 1).type_as(real_images)

        # 计算生成器的输出和判别器的预测
        fake_images = self(z)
        fake_preds = self.discriminator(fake_images)

        # 计算生成器损失
        val_g_loss = self.generator_loss(fake_preds)
        self.log("val_g_loss", val_g_loss, prog_bar=True)

        # 计算判别器的预测并计算损失
        real_preds = self.discriminator(real_images)
        val_d_loss = self.discriminator_loss(real_preds, fake_preds)
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
