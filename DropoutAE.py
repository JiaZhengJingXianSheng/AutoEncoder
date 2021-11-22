from torch import nn


class DropoutAE(nn.Module):
    def __init__(self):
        super(DropoutAE, self).__init__()
        self.encoder = nn.Sequential(
            # 32 * 32
            nn.Linear(784, 784),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Linear(784, 784),

            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(784, 784),

            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(784, 1),
            nn.ReLU()


        )
        self.decoder = nn.Sequential(

            nn.Linear(1, 784),
            nn.ReLU(),

            nn.Linear(784, 784),
            nn.ReLU(),

            nn.Linear(784, 784),
            nn.ReLU(),

            nn.Linear(784, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param [b, 1, 28, 28]:
        :return [b, 1, 28, 28]:
        """
        batchsz = x.size(0)
        # flatten
        x = x.view(batchsz, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        # reshape
        x = x.view(batchsz, 1, 28, 28)

        return x
