import matplotlib.pyplot as plt
import torch


def train(model, epochs, train_loader, val_loader, loss, optimizer):
    for epoch in range(epochs):
        model.to("cuda:0")
        model.train()
        running_loss = 0.0
        for i, (x, _) in enumerate(train_loader):
            x = x.to("cuda:0")
            x_hat = model(x)
            l = loss(x_hat, x)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            running_loss += l.item()
            rate = (i + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            lossEnd = running_loss / (i + 1)
            print("\rEpoch: {}  {:^3.0f}%[{}->{}] train loss: {:.5f}".format(epoch, int(rate * 100), a, b,
                                                                             lossEnd), end="")
        print()

        X, _ = iter(val_loader).next()
        X = X.to("cuda:0")
        with torch.no_grad():
            X_hat = model(X)

    showX = X.to("cpu")
    showXHat = X_hat.to("cpu")
    plt.figure(1)

    for i in range(1, 33):
        plt.subplot(4, 8, i)
        plt.imshow(showX[i - 1][0])
        plt.axis('off')
    plt.title('Original pics')

    plt.figure(2)

    for i in range(1, 33):
        plt.subplot(4, 8, i)
        plt.imshow(showXHat[i - 1][0])
        plt.axis('off')
    plt.title('AutoEncoder pics')
    plt.show()
