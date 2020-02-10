#!/usr/bin/env python3

"""
PyTorch example of predicting handwritten numbers

"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
from torchvision import datasets, transforms
from os import path

# two datasets: train & test
# built-in dataset: MNIST (pictures of handwritten numbers 28x28)
file_path = path.dirname(path.realpath(__file__))
train = datasets.MNIST(file_path, train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST(file_path, train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

trainingset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testingset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# looking at our data
for data in trainingset:
    # print(data)
    break

# since our data variable is two seperate tensors of tensors, we can seperate
# them based on the image and its real value
X, y = data[0][0], data[1][0]

# using matplotlib to view images in our dataset
# we have a problem, PyTorch wants [1, 28, 28] but we can only view 28x28 so we
# view the Tensor as 28x28
print(X.shape)
plt.imshow(X.view(28, 28))
# plt.show()

# in the best case scenario, you want an even amount of items per data value
"""
total = 0
counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
for data in trainingset:
    Xs, ys = data
    for y in ys:
        counter[int(y)] += 1
        total += 1
for i in counter:
    print(f'{i}: {round((counter[i] / total) * 100, 2)}%')
"""


# creating our neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # the first input is the size of our 'flattened' data
        # the last output is how many things we can classify the data as
        self.fc1 = nn.Linear(28*28, 64)  # fully connected 1
        self.fc2 = nn.Linear(64, 64)  # fully connected 2
        self.fc3 = nn.Linear(64, 64)  # fully connected 3
        self.fc4 = nn.Linear(64, 10)  # fully connected 4

    def forward(self, x):
        """ simple NN: 'FeedForward Network' """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


# lets test our NN with fake data
"""
net = Net()
X = torch.rand((28, 28))
X = X.view(-1, 28 * 28)
output = net(X)
print(output)
"""

# time to optimize our model
net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3  # amount of times we run through all our data
for epoch in range(EPOCHS):
    for data in trainingset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0
with torch.no_grad():
    for data in trainingset:
        X, y = data
        output = net(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print(f'Accuracy: {round(correct / total, 3)}')

# lets check it out
plt.imshow(X[0].view(28, 28))
plt.show()
print(torch.argmax(net(X[0].view(-1, 28*28))[0]))
