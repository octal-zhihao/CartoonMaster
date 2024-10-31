import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from .cartoon_dataset import CartoonDataset

class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_path = kwargs["data_path"]
        self.batch_size = kwargs["batch_size"]
        self.image_size = kwargs["image_size"]
        self.num_workers = kwargs["num_workers"]
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """加载并初始化数据集"""
        full_dataset = CartoonDataset(root_dir=self.data_path, 
                                      image_size=self.image_size)

        # 划分训练集和验证集（例如 80% 训练集，20% 验证集）
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        """返回训练数据的 DataLoader"""
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        """返回验证数据的 DataLoader"""
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers)
