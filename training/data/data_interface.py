import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .cartoon_dataset import CartoonDataset

class DInterface(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 64, image_size: int = 96):
        super(DInterface, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size

    def setup(self, stage=None):
        """加载并初始化数据集"""
        self.dataset = CartoonDataset(root_dir=self.data_path, image_size=self.image_size)

    def train_dataloader(self):
        """返回训练数据的 DataLoader"""
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
