#!/usr/bin/env python3

"""
PyTorch example of predicting whether something is a cat or a dog

"""

import os
import cv2
import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUILD_DATA = False  # set to true when generating training data otherwise load


class CatsVSDogs():
    IM_SIZE = 50
    CATS = 'PetImages/Cat'
    DOGS = 'PetImages/Dog'
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    cat_count = 0
    dog_count = 0

    def gen_train_data(self):
        for label in self.LABELS:
            print(label)
            # tqdm is a loading bar just so we can see prograss of for loop
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    im = cv2.resize(im, (self.IM_SIZE, self.IM_SIZE))
                    # 2d array with each sub-array having [image, vector]
                    self.training_data.append(
                        [np.array(im), np.eye(2)[self.LABELS[label]]]
                    )
                    if label == self.CATS:
                        self.cat_count += 1
                    elif label == self.DOGS:
                        self.dog_count += 1
                except Exception:
                    # some images in this dataset are currupt
                    pass

        # shuffle data so it doesnt learn to predict all cats and then all dogs
        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)  # save training data
        print(f'Cats: {self.cat_count}\tDogs: {self.dog_count}')


if BUILD_DATA:
    catsvdogs = CatsVSDogs()
    catsvdogs.gen_train_data()

training_data = np.load('training_data.npy', allow_pickle=True)
# print(len(training_data))
# plt.imshow(training_data[0][0], cmap='gray')
# plt.show(training_data[0][0].all())


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        # in order to find the inital input to the linear layers, we to run
        # 'fake data' through the conv layers and see what the output will be
        # so you can determine what the input will be of the linear layers
        # THERE IS AN torch.flatten(x) METHOD FOR THIS NOW
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            # this is the output of the convs
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


net = Net().double()

optimizer = optim.Adam(net.parameters(), lr=1e-3)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X / 255.0
y = torch.tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X) * VAL_PCT)
# print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]
test_X = X[-val_size:]
test_y = y[-val_size:]
# print(len(train_X))
# print(len(test_X))

BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        # print(i, i + BATCH_SIZE)
        batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i + BATCH_SIZE]

        net.zero_grad()
        outputs = net(batch_X.double())
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

print(loss)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_value = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50).double())[0]
        predicted_value = torch.argmax(net_out)
        if predicted_value == real_value:
            correct += 1
        total += 1
print(f'Accuracy: {round(correct / total, 2)}')
