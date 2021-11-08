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


class Dataset(Dataset):
    def __init__(self, csv_file):
        raw_data = pd.read_csv(csv_file, header=None)
        self.features = torch.from_numpy(np.array(raw_data.iloc[:, 0:16])).float().to(device)
        self.labels = torch.from_numpy(np.array(raw_data.iloc[:, 16])).to(device)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)


test_data = Dataset("data/pendigits.tes")


test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)


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


#model = NeuralNetwork().to(device)
model = torch.load(f"./models/epo_2021-11-07-06-44-02")
criterion = nn.CrossEntropyLoss()


def test_loop(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error|| Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    test_loop(test_dataloader, model, criterion)
print("Done!")
