import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data.data_interface import DInterface
from models.model_interface import MInterface
from models.Unrolled_model_interface import UnrolledMInterface
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
import wandb

def predict_demo(model, latent_dim, save_dir="generated_images"):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成随机噪声
    noise = torch.randn(5, latent_dim)  # 生成 5 个随机噪声，形状为 [5, latent_dim]
    
    # 将噪声张量的形状调整为 [5, latent_dim, 1, 1] 以适配生成器的输入
    noise = noise.view(5, latent_dim, 1, 1)
    device = next(model.parameters()).device
    noise = noise.to(device)
    # 生成图像
    generated_images = model.generator(noise)
    print(generated_images.size())
    # 处理生成的图像（如保存或展示）
    for i in range(generated_images.size(0)):
        # 将图像保存
        save_image(generated_images[i], os.path.join(save_dir, f"generated_image_{i}.png"))

    print(f"Generated images saved to {save_dir}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN for cartoon image generation")
    parser.add_argument("--data_path", type=str, default="./dataset/cartoon_color", help="Path to cartoon image dataset")
    parser.add_argument("--image_size", type=int, default=128, help="Size of the input image to the model")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loaders")
    parser.add_argument("--g_lr", type=float, default=0.0003, help="Generator learning rate")
    parser.add_argument("--d_lr", type=float, default=0.0001, help="Discriminator learning rate")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, default="Unrolled_DCGAN", help="Name of the model to train")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run the script in: train or predict")
    parser.add_argument("--checkpoint_path", type=str, default="my_checkpoint/epoch=29-step=15300.ckpt", help="Path to the model checkpoint for predictions")
    args = parser.parse_args()

    

    if args.mode == "predict":
        # 初始化数据模块和模型
        data_module = DInterface(**vars(args))
        if args.model_name == "Unrolled_DCGAN":
            model = UnrolledMInterface.load_from_checkpoint(**vars(args))
        else:
            model = MInterface.load_from_checkpoint(**vars(args))
        model.eval()
        # 进行预测
        predict_demo(model, args.latent_dim, save_dir="generated_images")
        exit()
    else:
        wandb.init(project="cartoon-gan", config=vars(args))
        wandb_logger = WandbLogger(project="cartoon-gan", config=vars(args))
        # 初始化数据模块和模型
        data_module = DInterface(**vars(args))
        if args.model_name == "Unrolled_DCGAN":
            model = UnrolledMInterface(**vars(args))
        else:
            model = MInterface(**vars(args))
        # 创建 ModelCheckpoint 回调
        checkpoint_callback = ModelCheckpoint(
            monitor='val_g_loss',                # 监控验证集损失
            save_top_k=-1,                   # 保存所有检查点
            every_n_epochs=10,                # 每10个epoch保存一次
            mode='min'                       # 最小化监控指标
        )


        # 训练
        trainer = Trainer(
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback],
            logger=wandb_logger
        )
        trainer.fit(model, data_module)
