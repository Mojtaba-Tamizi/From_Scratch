import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'image')
        self.mask_dir = os.path.join(root_dir, 'mask')
        self.images = os.listdir(self.image_dir)
        self.masks = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_name)
        mask = Image.open(mask_name)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Ensure mask is a tensor of integers
        mask = torch.tensor(np.array(mask), dtype=torch.int64)

        return image, mask

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

dataset_train = SegmentationDataset(root_dir='../Datasets/Retina_Blood_Vessel/train/', transform=transform)
dataset_test = SegmentationDataset(root_dir='../Datasets/Retina_Blood_Vessel/test/', transform=transform)

# Create a DataLoader instance
dataloader_train = DataLoader(dataset_train , batch_size=4, shuffle=True)
dataloader_test = DataLoader(dataset_test , batch_size=8, shuffle=False)

def plot():
    image, mask = dataset_test[0]  # Change index to plot different images

    # Convert image and mask tensor to numpy arrays (transpose if necessary)
    image = image.permute(1, 2, 0).numpy()  # Assuming image is tensor of shape (C, H, W)
    mask = mask.squeeze().numpy() # Assuming mask is tensor of shape (H, W)

    # Plot the image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')  # Assuming mask is grayscale
    plt.title('Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

plot()