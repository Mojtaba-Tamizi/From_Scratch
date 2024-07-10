import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from UNet import UNet
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
dataloader_train = DataLoader(dataset_train , batch_size=2, shuffle=True)
dataloader_test = DataLoader(dataset_test , batch_size=4, shuffle=False)

def plot(image, mask, predict, epoch):
    # Convert image and mask tensor to numpy arrays (transpose if necessary)
    image = image.permute(1, 2, 0).numpy()  # Assuming image is tensor of shape (C, H, W)
    mask = mask.squeeze().numpy() # Assuming mask is tensor of shape (H, W)
    predict = predict.squeeze().numpy()
    # Plot the image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')  # Assuming mask is grayscale
    plt.title('Mask')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(predict, cmap='gray')  # Assuming mask is grayscale
    plt.title('predict')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'results_{epoch+1}.png')

def test_every_epoch(epoch):
    image, mask = dataset_test[0]
    model.eval()
    with torch.no_grad():
        predict = model(image.unsqueeze(0).to(DEVICE).float()).to('cpu')

    plot(image, mask, predict.squeeze(), epoch)

model = UNet(in_channels=3, classes=1).to(DEVICE)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = StepLR(optimizer, step_size=25, gamma=0.3)
m = nn.Sigmoid()
num_epochs = 100
for epoch in tqdm(range(num_epochs)):
    model.train()
    for images, masks in dataloader_train:  # Assume you have a DataLoader `train_loader`
        optimizer.zero_grad()
        outputs = m(model(images.to(DEVICE)).float())
        masks = masks.float().to(DEVICE)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    scheduler.step()
    if epoch % 10 == 0:
        test_every_epoch(epoch)
    current_lr = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, learning_rate:{current_lr}')
test_every_epoch(100)
torch.save(model.state_dict(), 'unet_model_100epochs.pth')

