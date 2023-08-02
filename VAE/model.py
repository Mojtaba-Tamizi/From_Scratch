import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAuroEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20) -> None:
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder      
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        self.relu = nn.LeakyReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)

        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))

        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.rand_like(sigma)
        z_repara = mu + sigma*epsilon

        x_recon = self.decode(z_repara)

        return x_recon, mu, sigma

if __name__ == '__main__':
    x = torch.randn(4, 784) # 28 * 28 for MNIST dataset
    vae = VariationalAuroEncoder(input_dim=784)
    x_recon, mu, sigma = vae(x)
    print(f'x_recon shape : {x_recon.shape}')
    print(f'mu shape : {mu.shape}')
    print(f'sigma shape : {sigma.shape}')
