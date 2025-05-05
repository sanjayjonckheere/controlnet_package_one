"""Dataset handling for ControlNet training."""

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os
from typing import Tuple
import torch
import random


class DepthDataset(Dataset):
    """Dataset for loading paired images and depth maps with augmentation."""
    def __init__(self, data_root: str, image_size: int):
        self.data_root = data_root
        self.image_size = image_size
        self.image_files = sorted([f for f in os.listdir(data_root) if f.startswith('image_') and f.endswith('.jpg')])
        self.depth_files = sorted([f for f in os.listdir(data_root) if f.startswith('depth_') and f.endswith('.png')])
        if len(self.image_files) != len(self.depth_files):
            raise ValueError(f"Mismatch in number of images ({len(self.image_files)}) and depth maps ({len(self.depth_files)})")
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.depth_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self) -> int:
        return len(self.image_files)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.data_root, self.image_files[idx])
        depth_path = os.path.join(self.data_root, self.depth_files[idx])
        image = Image.open(img_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        random.seed(seed)
        torch.manual_seed(seed)
        depth = self.depth_transform(depth)
        return image, depth
