import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt # Visualize data
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # OMP: Error #15 해결

import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 300
BATCH_SIZE = 128

transform = transforms.Compose([
transforms.ToTensor()
])

trainset = datasets.CIFAR100(
    root = './.data/', 
    train = True,
    download = True,
    transform = transform
    )

testset = datasets.CIFAR100(
    root = './.data/', 
    train = False,
    download = True,
    transform = transform
    )

train_loader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    )

test_loader = torch.utils.data.DataLoader(
    dataset = testset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3072, 1024) # 3체널 32x32 데이터 1차원으로 reshape
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 100) # 100 classese

    def forward(self, x):
        x = x.view(-1, 3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE) # 학습 데이터를 DEVICE의 메모리로 보냄
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    return loss        # plot 위해 return
    

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            # 모든 오차 더하기
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # 가장 큰 값을 가진 클래스가 모델의 예측입니다.
            # 예측과 정답을 비교하여 일치할 경우 correct에 1을 더합니다.
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


train_loss_list = []
test_loss_list = []

for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader, optimizer)
    train_loss_list.append(train_loss.cpu()) # tensor -> numpy
    test_loss, test_accuracy = evaluate(model, test_loader)
    test_loss_list.append(test_loss)
    print(f'test loss : {test_loss}')
torch.save(model.state_dict(), 'DNN_CIFAR100/dnn_model.pt')

# plot loss
train_loss_list = [loss.detach() for loss in train_loss_list]
plt.plot(range(1,EPOCHS+1), np.array(train_loss_list),'r', label = 'train_loss')
plt.plot(range(1,EPOCHS+1), np.array(test_loss_list),'b', label = 'test_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()