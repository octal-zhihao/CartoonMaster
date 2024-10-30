import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .cartoon_dataset import CartoonDataset

class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_path = kwargs["data_path"]
        self.batch_size = kwargs["batch_size"]
        self.image_size = kwargs["image_size"]
        self.num_workers = kwargs["num_workers"]

    def setup(self, stage=None):
        """加载并初始化数据集"""
        self.dataset = CartoonDataset(root_dir=self.data_path, 
                                      image_size=self.image_size)

    def train_dataloader(self):
        """返回训练数据的 DataLoader"""
        return DataLoader(self.dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)
