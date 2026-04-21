import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class PointCloudMNIST(Dataset):
    def __init__(self, root='./data', train=True, num_points=256, threshold=0.1):
        """
        Downloads MNIST and converts it to 3D point clouds on the fly.
        num_points: The exact 'N' dimension for your (B, N, 3) tensor.
        threshold: Pixel intensity to consider 'solid'.
        """
        super().__init__()
        self.num_points = num_points
        self.threshold = threshold
        self.mnist = datasets.MNIST(root=root, train=train, download=True, transform=transforms.ToTensor()) 
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx: int):
        img, label = self.mnist[idx]
        img = img.squeeze(0).numpy() 
        y_coords, x_coords = np.where(img > self.threshold)
        
        if len(y_coords) == 0:
            return torch.zeros(self.num_points, 3), label
    
        x = torch.from_numpy(x_coords).float()
        y = torch.from_numpy(y_coords).float()
        x = (x / 27.0) * 2.0 - 1.0
        y = -(y / 27.0) * 2.0 + 1.0
        
        z = torch.zeros_like(x)
        pc = torch.stack([x, y, z], dim=1)
        num_active_pixels = pc.shape[0]
        
        if num_active_pixels >= self.num_points:
            indices = torch.randperm(num_active_pixels)[:self.num_points]
        else:
            indices = torch.randint(0, num_active_pixels, (self.num_points,))
            
        pc_fixed = pc[indices]
        
        return pc_fixed, label