import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import psutil
import matplotlib.pyplot as plt



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2000, 900), 
            nn.ReLU(),
            nn.Linear(900,250), 
            nn.ReLU(),
            nn.Linear(250,50),
            nn.ReLU(),
            nn.Linear(50, 2) 
        )

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

class AvgMeter():
    sum: float
    count: int

    def __init__(self) -> None:
        self.sum = 0.
        self.count = 0

    def append(self, sum: float, count: int = 0):
        self.sum += sum
        self.count += count
    
    @property
    def avg(self) -> float:
        return self.sum / self.count
        

pth = 'Data/tomo_train_240608/train/'
train_filename = {'g': ['train_ground.csv'], 'e': ['train_excited.csv']}
# train_filename = {'g': ['HF_ground.csv'], 'e': ['HF_excited.csv']}

# Create (key, value) pair
chara_lst = []; label_lst = []
for keys in train_filename.keys():
    for filename in train_filename[keys]:
        data = pd.read_csv(pth+filename, header=None)
        chara_set=data.iloc[:,:].to_numpy().astype(float)
        if keys == 'e':
            label_set=np.ones(data.shape[0], dtype=int).T
        elif keys == 'g':
            label_set=np.zeros(data.shape[0], dtype=int).T
        else:
            raise(ValueError(f'Invalid keys {keys}'))
        chara_lst.append(chara_set)
        label_lst.append(label_set)
chara_arr = np.vstack(chara_lst); label_arr = np.hstack(label_lst)

X_train, X_eval, Y_train, Y_eval = train_test_split(chara_arr, label_arr, test_size=0.2, random_state=3)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
trainset = TensorDataset(X_train_tensor, Y_train_tensor)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
evalset = TensorDataset(torch.tensor(X_eval, dtype=torch.float32), torch.tensor(Y_eval, dtype=torch.long))
evalloader = DataLoader(evalset)


def train_mlp(trainloader: DataLoader, evalloader: DataLoader, learning_rate= 0.005, epochs= 150):
    """
    learning_rate: float
        value between 0 and 1

    epochs: int
        number of training iterations.
    """
    config = {'lr': learning_rate, 'epochs': epochs}
    mlp = MLP()
    criterion = nn.CrossEntropyLoss(reduction='sum')  # Using cross entropy function for muti-classification problem
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)  # Adam optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # whether use GPU or not

    for i in range(epochs):
        running_loss = AvgMeter()
        running_acc = AvgMeter()
        # running_loss = 0.
        # running_acc = 0.

        for (inputs, labels) in trainloader:
            batch_size = inputs.size(0)

            optimizer.zero_grad()
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item(), count=batch_size)
            pred = torch.argmax(outputs, dim=1)
            assert pred.shape == labels.shape

            running_acc.append((torch.argmax(outputs, dim=1) == labels).sum(), count=batch_size)
            # running_loss += loss.item()
            # _, predict = torch.max(outputs, 1) # arg max
            # correct_num = (predict == labels).sum()
            # running_acc += correct_num.item()
        

        # running_loss /= len(trainset)
        # running_acc /= len(trainset)
        if (i+1) % 10 ==0:
            print(f"[{i+1}/{epochs}] Loss:{running_loss.avg:.5f}, Acc:{eval_model(mlp, evalloader):.2f}")

    return mlp, config


def eval_model(model: MLP, evalloader: DataLoader) -> float:
    with torch.no_grad():
        acc = AvgMeter()
        for (x, y) in evalloader:
            assert x.shape[0] == 1
            y_hat = torch.argmax(model(x), dim=1)
            acc.append((y_hat == y).sum(), count = x.shape[0])
    return acc.avg
    

def inference(model: MLP, X_test: torch.Tensor):
    with torch.no_grad():
        y_hat = torch.argmax(model(X_test), dim=1)
        assert X_test.shape[0] == y_hat.shape[0]
        return y_hat