import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import logging
import os.path
import urllib
from torch.optim.lr_scheduler import LambdaLR
import time

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

device = 'cpu'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Using {} device'.format(device))

if os.path.exists("./data/") is False:
    os.mkdir("./data/")
if os.path.exists("./models/") is False:
    os.mkdir("./models/")

if os.path.exists("./data/pendigits.tra") is False and os.path.exists("./data/pendigits.tes") is False:
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra", "./data/pendigits.tra")
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes", "./data/pendigits.tes")


class Dataset(Dataset):
    def __init__(self, csv_file):
        raw_data = pd.read_csv(csv_file, header=None)
        self.features = torch.from_numpy(np.array(raw_data.iloc[:, 0:16])).float().to(device)
        self.labels = torch.from_numpy(np.array(raw_data.iloc[:, 16])).to(device)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)


training_data = Dataset("data/pendigits.tra")
test_data = Dataset("data/pendigits.tes")


train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


model = NeuralNetwork().to(device)


def train_loop():
    for X, y in train_dataloader:
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    # if batch % 100 == 0:
    #     loss, current = loss.item(), batch * len(X)
    #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop():
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error|| Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epoch = 5
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch/10+11))

for epoch in range(200):
    train_loop()
    if(epoch % 100 == 0):
        print(f"Epoch {epoch+1}\n-------------------------------")
        print(f"lr: {optimizer.param_groups[0]['lr']}")
        test_loop()

curtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
torch.save(model, f"./models/epo_{curtime}")
print("Done!")
