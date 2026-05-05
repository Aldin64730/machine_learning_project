import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF
import random

class SODDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(224, 224), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.augment = augment
        
        # Sort to ensure images match masks verbatim
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def transform(self, image, mask):
        # 1. Resize: Standardize to 224x224
        image = TF.resize(image, self.target_size)
        mask = TF.resize(mask, self.target_size)

        if self.augment:
            # 2. Augmentation: Horizontal Flip (Sync'd)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 3. Augmentation: Brightness (Image only)
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)

        # 4. Normalize: Convert to Tensor and scale [0, 1]
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        return image, mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask
        
        image, mask = self.transform(image, mask)
        return image, mask

def get_dataloaders(batch_size=16):
    # Adjusting paths based on your /src structure
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "DUTS", "DUTS-TR"))
    img_dir = os.path.join(base_path, "DUTS-TR-Image")
    mask_dir = os.path.join(base_path, "DUTS-TR-Mask")

    full_dataset = SODDataset(img_dir, mask_dir, augment=True)
    
    # Split into Train (70%), Validation (15%), and Test (15%)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Quick test block
if __name__ == "__main__":
    train_l, _, _ = get_dataloaders()
    imgs, msks = next(iter(train_l))
    print(f"Batch Image Shape: {imgs.shape}") # Expect [16, 3, 224, 224]
    print(f"Batch Mask Shape: {msks.shape}")   # Expect [16, 1, 224, 224]