import torch
from torch import nn


class MODEL(nn.Module):
    """CNN VAE network"""

    def __init__(self, hyper_params):
        super(MODEL, self).__init__()
        self.hpm = hyper_params
        self.CNN_shape = None  # record the shape of batch matrix after CNN
        self.enc_CNN = nn.Sequential(  # (B, 1, 28, 28)
            nn.Conv2d(1, 16, 3, 3, 1),  # (B, 16, 10, 10)
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 16, 5, 5)
            nn.Conv2d(16, 8, 3, 2, 1),  # (B, 8, 3, 3)
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 1)  # (B, 8, 2, 2)
        )
        self.fc_mu = nn.Linear(8*2*2, 16)
        self.fc_logvar = nn.Linear(8*2*2, 16)
        self.fc_z = nn.Linear(16, 8*2*2)
        self.dec_CNN = nn.Sequential(  # (B, 8, 2, 2)
            nn.ConvTranspose2d(8, 16, 3, 2),  # 2*3-2+1=5, (B, 16, 5, 5)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, 3, 1),  # 5*5-(5-1)*2-1*2=15, (B, 8, 15, 15)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 2, 2, 1),  # 2*15-1*2=28, (B, 1, 28, 28)
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5*logvar)
            # std = torch.exp(0.5*logvar)  # 0.5 mul is a must
            # eps = torch.randn_like(std)
            # return mu + eps*std
            # return torch.normal(mu, torch.exp(0.5 * logvar))
        else:
            return mu

    def encode(self, x):
        # print("Encode:", x.shape, end=" -> ")
        x = self.enc_CNN(x)
        # print(x.shape, end=" -> ")
        self.CNN_shape = x.shape
        x = x.view(x.size(0), -1)
        # print(x.shape)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z):
        # print("Decode: ", z.shape, end=" -> ")
        z = self.fc_z(z).view(self.CNN_shape)
        # print(z.shape)
        return self.dec_CNN(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss_func(recon_x, x, mu, logvar, BCE_loss=nn.MSELoss(reduction="sum"), beta=1):
        BCE = BCE_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD

