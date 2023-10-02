from torch import nn
import torch

class PrintShapeSequential(nn.Sequential):
    def forward(self, inp):
        for module in self:
            inp = module(inp)
            print(inp.shape)
        return inp



# what happens if you significantly reduce
# expressivity of Encoder block? 
# ... stunted works just as well as dnn
class Encoder(nn.Module):
    def __init__(self, z_dim, dropout=0.1):
        super().__init__()
        self.stunted =  nn.Sequential(
            nn.Dropout1d(p=0.1),
            nn.Linear(784,128),
            nn.Sigmoid()
        )
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
            self.stunted,
            nn.Linear(128,z_dim)
        )
        self.variance = nn.Sequential(
            self.stunted,
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
        self.z_dim = z_dim
    
    def forward(self, x):
        z = self.encoder(x)
        #print(z.shape)
        o = self.decoder(z)
        return o

class CEncoder(nn.Module):
    def __init__(self, z_dim, dropout=0.1):
        super().__init__()
        self.stunted =  nn.Sequential(
            nn.Dropout1d(p=0.1),
            nn.Linear(794,128),
            nn.Sigmoid()
        ) 
        self.dnn = nn.Sequential(
            # 26x26 # target 512
            nn.Dropout1d(p=dropout),
            nn.Linear(794,512),
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

    def forward(self,x,y):
        x = torch.cat((x,y),dim=-1)
        m,v = self.mean(x), self.variance(x)
        z = self.reparameterize(m,v)
        return z, m, v
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

class CDecoder(nn.Module):
    def __init__(self,z_dim, dropout=0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            #  latent_dim -> 128
            nn.Linear(z_dim+10,128),
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
    def forward(self, z, y):
        ret = self.decoder(torch.cat((z,y), dim=-1))
        return ret

class CVAE(nn.Module):
    def __init__(self, z_dim, dropout=0.1):
        super().__init__() 
        self.encoder = CEncoder(z_dim, dropout=dropout)
        self.decoder = CDecoder(z_dim, dropout=dropout)
    
    def forward(self,x,y):
        z,m,v = self.encoder(x,y)
        recon_x = self.decoder(z,y)
        return recon_x,m,v
