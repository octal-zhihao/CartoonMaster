import argparse
import torch
from pytorch_lightning import Trainer
from data.data_interface import DInterface
from models.model_interface import MInterface

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
    model = MInterface(latent_dim=args.latent_dim, lr=args.lr)

    # 训练
    trainer = Trainer(max_epochs=args.epochs)
    trainer.fit(model, data_module)