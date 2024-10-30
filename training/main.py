import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from data.data_interface import DInterface
from models.model_interface import MInterface
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def predict_demo(model, latent_dim, save_dir="generated_images"):
    # 生成随机噪声
    noise = torch.randn(5, latent_dim)  # 生成 5 个随机噪声，形状为 [5, latent_dim]
    
    # 将噪声张量的形状调整为 [5, latent_dim, 1, 1] 以适配生成器的输入
    noise = noise.view(5, latent_dim, 1, 1)
    
    # 生成图像
    generated_images = model.generator(noise)

    # 处理生成的图像（如保存或展示）
    for i in range(generated_images.size(0)):
        # 将图像保存或显示
        save_image(generated_images[i], f"{save_dir}/generated_image_{i}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN for cartoon image generation")
    parser.add_argument("--data_path", type=str, default="./dataset/cartoon_color", help="Path to cartoon image dataset")
    parser.add_argument("--image_size", type=int, default=96, help="Size of the input image to the model")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loaders")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, default="DC_GAN", help="Name of the model to train")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run the script in: train or predict")
    parser.add_argument("--checkpoint_path", type=str, default="lightning_logs/version_2/checkpoints/epoch=1-step=640.ckpt", help="Path to the model checkpoint for predictions")
    args = parser.parse_args()

    if args.mode == "predict":
        # 初始化数据模块和模型
        data_module = DInterface(**vars(args))
        model = MInterface.load_from_checkpoint(**vars(args))
        model.eval()
        # 进行预测
        predict_demo(model, args.latent_dim, save_dir="generated_images")
        exit()
    else:
        # 初始化数据模块和模型
        data_module = DInterface(**vars(args))
        model = MInterface(**vars(args))
        # 训练
        trainer = Trainer(max_epochs=args.epochs)
        trainer.fit(model, data_module)
