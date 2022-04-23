import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


class FC_LYR(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, act_func: callable = None):
        """
        activation function after the layer operation
        if the activation function is not assigned, use default None
        """
        super(FC_LYR, self).__init__()
        if act_func is not None:
            self.fc = nn.Sequential(
                nn.Linear(ch_in, ch_out),
                act_func(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(ch_in, ch_out),
            )

    def forward(self, x):
        return self.fc(x)
        # print(x.shape, end=" -> ")
        # x = self.fc(x)
        # print(x.shape)
        # return x


class MODEL(nn.Module):
    """
    neural network based on MLP
    """
    def __init__(self, hyper_params):
        super(MODEL, self).__init__()
        self.hpm = hyper_params
        self.act_func = nn.ELU
        self.enc_MLP = nn.Sequential(OrderedDict(
            [("enc_MLP_"+str(idx), FC_LYR(ch_in, ch_out, self.act_func)) for idx, (ch_in, ch_out) in enumerate(zip(self.hpm["enc_MLP"][:-2], self.hpm["enc_MLP"][1:-1]))]
        ))
        self.enc_mu = nn.Sequential(
            FC_LYR(self.hpm["enc_MLP"][-2], self.hpm["enc_MLP"][-1], None),
        )
        self.enc_logvar = nn.Sequential(
            FC_LYR(self.hpm["enc_MLP"][-2], self.hpm["enc_MLP"][-1], None),
        )
        self.dec_MLP = nn.Sequential(OrderedDict(
            [("dec_MLP_"+str(idx), FC_LYR(ch_in, ch_out, self.act_func)) for idx, (ch_in, ch_out) in enumerate(zip(self.hpm["dec_MLP"][:-2], self.hpm["dec_MLP"][1:-1]))]
            + [("dec_MLP_"+str(len(self.hpm["dec_MLP"])-2), FC_LYR(self.hpm["dec_MLP"][-2], self.hpm["dec_MLP"][-1], nn.Sigmoid))]
        ))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)  # return a normal distribution tensor
            return mu + eps*std
            # return mu + torch.normal(mean = mu, std = torch.exp(0.5*logvar))
        else:
            return mu

    def forward(self, x):
        x_shape = x.shape
        x = self.enc_MLP(x.view(x.size(0), -1))
        mu, logvar = self.enc_mu(x), self.enc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.dec_MLP(z).view(x_shape)
        return x, mu, logvar, z

    @staticmethod
    def loss_func(recon_x, x, mu, logvar, beta=1.0):
        # BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
        BCE = F.mse_loss(recon_x, x, reduction="sum")  # use l2 / MSE loss to replace BCE loss
        # BCE = F.l1_loss(recon_x, x, reduction="sum")  # use l1 / MAE loss to replace BCE loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD

