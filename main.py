import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim
from AE import AE
from Train import train

train_dataset = datasets.MNIST('mnist', train=True, transform=transforms.Compose([transforms.ToTensor()]),
                               download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.MNIST('mnist', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
val_loader = DataLoader(val_dataset, batch_size=32)

epochs = 1
lr = 0.001
model = AE()
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
print(model)
device = torch.device("cuda:0")

if __name__ == '__main__':
    train(model, epochs, train_loader, val_loader, loss, optimizer)
