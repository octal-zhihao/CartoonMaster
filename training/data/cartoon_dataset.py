import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CartoonDataset(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.root_dir = root_dir
        self.image_size = image_size
        self.images = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(".jpg")]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)  # 归一化到[-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
