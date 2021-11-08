import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import logging
import os.path
import urllib

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = torch.device('cuda')
device = torch.device('cpu')
logger.info('Using {} device'.format(device))

if os.path.exists("./data/") is False:
    os.mkdir("./data/")
if os.path.exists("./models/") is False:
    os.mkdir("./models/")

if os.path.exists("./data/pendigits.tra") is False and os.path.exists("./data/pendigits.tes") is False:
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra", "./data/pendigits.tra")
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes", "./data/pendigits.tes")


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.raw_data = pd.read_csv(csv_file, header=None)
        self.features = torch.from_numpy(np.array(self.raw_data.iloc[:, 0:16])).float().to(device)

        origin_labels = torch.from_numpy(np.array(self.raw_data.iloc[:, 16:17])).to(device)
        len = origin_labels.shape[0]
        temp = np.zeros([len, 10])

        for i in range(len):
            temp[i][origin_labels[i]] = 1

        self.labels = torch.from_numpy(temp).float().to(device)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        #assert len(self.features) == len(self.labels)
        return len(self.features)


training_data = MyDataset("data/pendigits.tra")
test_data = MyDataset("data/pendigits.tes")


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y.argmax(dim=1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error|| Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 2000
for t in range(epochs):
    train_loop(train_dataloader, model, loss_fn, optimizer)
    if(t % 50 == 0):
        print(f"Epoch {t+1}\n-------------------------------")
        test_loop(test_dataloader, model, loss_fn)
print("Done!")
