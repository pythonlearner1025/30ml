import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.functional import F
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from mygrad.utils import *
from vae import *
import os
import argparse
import random

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "cpu"
else:
    DEVICE = "cpu"

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

dig2label = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
label2dig = {v:k for k,v in dig2label.items()}

def loss_function(recon_x, x, mu, logvar):
    BCE = F.cross_entropy(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(save_path, c=False, dropout=0.1, z_dim=64, lr=0.001, batch_size=64, epochs=5):
    if c:
        print("Training Conditional VAE")
        model = CVAE(z_dim,dropout=dropout)
    else:
        print("Training VAE")
        model = VAE(z_dim, dropout=dropout)

    model.to(DEVICE)

    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for i in range(epochs):
        for batch, (X,Y) in enumerate(train_dataloader):
            X,Y = X.to(DEVICE), Y.to(DEVICE)
            X = X.view(X.shape[0],-1)
            if c:
                Y = torch.tensor([one_hot(y, 10) for y in Y]).to(DEVICE)
                pred,m,v = model(X, Y)
            else:
                pred,m,v = model(X)

            loss = loss_fn(pred, X)
            #loss = loss_function(pred,X,m,v)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0: 
                loss, current = loss.item(), (batch+1) * len(X)
                size = len(train_dataloader.dataset)
                print(f"epoch {i} | loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        torch.save(model.decoder.state_dict(), save_path+f'{i}.pth')

def test(path, c=False, z_dim=64):
    if c:
        print("Testing Conditional VAE")
        decoder = CDecoder(z_dim)        
    else:
        print("Testing VAE")
        decoder = Decoder(z_dim)
    decoder.load_state_dict(torch.load(path))
    decoder.eval()

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 4, 4
    for i in range(1, cols*rows+1):
        z = torch.randn((1,z_dim))
        dig = 0
        dig = random.randint(0,9)
        hot = torch.tensor(one_hot(dig,10)).view(1,10)
        img = decoder(z,hot).view(1,28,28).detach().numpy()
        figure.add_subplot(rows, cols, i)
        plt.title(dig2label[dig])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--conditional", type=int, required=False)
    args = parser.parse_args()
    print(f'Using device: {DEVICE}')
    conditional = True if args.conditional != 0 else False

    cwd = os.getcwd()
    save_path = os.path.join(cwd, "weights/decoder_at_epoch_")
    # data = 28x28x3 tensor, and classificaton value (ignore)
    z_dim=10
    train(save_path, c=conditional, z_dim=z_dim,dropout=0.1, epochs=2)
    last = sorted(os.listdir(os.path.join(cwd, "weights")),key=lambda x:int(x[-5]))[-1]
    test(os.path.join(cwd, 'weights', last),c=conditional,z_dim=z_dim)




   