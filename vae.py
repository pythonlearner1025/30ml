from torch import nn
import torch

class PrintShapeSequential(nn.Sequential):
    def forward(self, inp):
        for module in self:
            inp = module(inp)
            print(inp.shape)
        return inp

class Encoder(nn.Module):
    def __init__(self, z_dim, dropout=0.1):
        super().__init__()
        self.dnn = nn.Sequential(
            # 26x26 # target 512
            nn.Dropout1d(p=dropout),
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),

            # 24x24 # target 256
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            # 12x12 # target 128
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.Linear(128,128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
        )
        self.mean = nn.Sequential(
            self.dnn,
            nn.Linear(128,z_dim)
        )
        self.variance = nn.Sequential(
            self.dnn,
            nn.Linear(128,z_dim)
        )
        self.e = torch.randn((1, z_dim))
    
    def forward(self, x):
        u,v = self.mean(x), self.variance(x)
        z = v * self.e + u
        return z


class Decoder(nn.Module):
    def __init__(self,z_dim, dropout=0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            #  latent_dim -> 128
            nn.Linear(z_dim,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # target 256
            nn.Dropout1d(p=dropout),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # target 512
            nn.Dropout1d(p=dropout),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,784),
            nn.Sigmoid(),
        )

    # let z be equal to x 
    def forward(self, z):
        ret = self.decoder(z)
        #print(ret.shape)
        return ret


class VAE(nn.Module):
    def __init__(self, z_dim, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(z_dim, dropout=dropout)
        self.decoder = Decoder(z_dim, dropout=dropout)
    
    def forward(self, x):
        z = self.encoder(x)
        o = self.decoder(z)
        return o

