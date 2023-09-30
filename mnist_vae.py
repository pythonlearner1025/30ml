import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from vae import *
import os

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

def train(save_path, dropout=0.1, z_dim=64, lr=0.001, batch_size=32, epochs=5):
    model = VAE(z_dim,dropout=dropout)

    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for i in range(epochs):
        for batch, (X,_) in enumerate(train_dataloader):
            X = X.view(X.shape[0],-1)
            pred = model(X)
            loss = loss_fn(pred, X)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0: 
                loss, current = loss.item(), (batch+1) * len(X)
                size = len(train_dataloader.dataset)
                print(f"epoch {i} | loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        torch.save(model.decoder.state_dict(), save_path+f'{i}.pth')

def test(path,z_dim=64):
    decoder = Decoder(z_dim)        
    decoder.load_state_dict(torch.load(path))
    decoder.eval()

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 4, 4
    for i in range(1, cols*rows+1):
        z = torch.randn((1,z_dim))
        img = decoder(z).view(1,28,28).detach().numpy()
        figure.add_subplot(rows, cols, i)
        plt.title("Gen Fashion")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

if __name__ == '__main__':
    cwd = os.getcwd()
    save_path = os.path.join(cwd, "weights/decoder_at_epoch_")
    # data = 28x28x3 tensor, and classificaton value (ignore)
    z_dim=64
    train(save_path,z_dim=z_dim,dropout=0.05)
    last = sorted(os.listdir(os.path.join(cwd, "weights")),key=lambda x:int(x[-5]))[-1]
    test(os.path.join(cwd, 'weights', last),z_dim=z_dim)




   