import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
from data_utils import get_data
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

device = 'cpu'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Using {} device'.format(device))


class Dataset(Dataset):
    def __init__(self, train=True, corrupt=False):
        features, labels = get_data(train, corrupt)
        # if corrupt == True:
        #     features, labels = ros.fit_resample(features, labels)
        self.features = torch.from_numpy(features).float().to(device)
        self.labels = torch.from_numpy(labels).to(device)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)


train_dataloader = DataLoader(Dataset(train=True, corrupt=False), batch_size=32, shuffle=True)
test_dataloader = DataLoader(Dataset(train=False, corrupt=False), batch_size=32, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def train_loop(model, criterion, optimizer):
    for X, y in train_dataloader:
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # scheduler.step()

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
    accuracy = 100*correct / size
    print(f"Test Error|| Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
    return accuracy, test_loss


# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-5)
# scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch/10+11))

torch.manual_seed(114514)
epochs = 30
epochs_list = np.arange(0, epochs)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_ylim(0, 100)
ax2.set_ylim(0, 100)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax2.set_ylabel("Test Loss")


def one_draw(model, criterion, optimizer, name, color):
    accuracy_list = []
    test_loss_list = []
    for epoch in range(epochs):
        train_loop(model, criterion, optimizer)
        # if(epoch % 20 == 0):
        print(f"Epoch {epoch+1}\n-------------------------------")
        # print(f"lr: {optimizer.param_groups[0]['lr']}")
        accuracy, test_loss = test_loop()
        accuracy_list.append(accuracy)
        test_loss_list.append(test_loss)

    ax1.plot(epochs_list, accuracy_list, label=name, color=color)
    ax2.plot(epochs_list, test_loss_list, label=name, color=color)


model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
one_draw(model, criterion, optimizer, "lr=1e-3", "green")

model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
one_draw(model, criterion, optimizer, "lr=1e-4", "blue")

model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
one_draw(model, criterion, optimizer, "lr=1e-5", "cyan")

plt.title("Adam and its hyper-parameters")
plt.legend()
plt.show()

# curtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
# torch.save(model, f"./models/epo_{curtime}")
# print("Done!")
