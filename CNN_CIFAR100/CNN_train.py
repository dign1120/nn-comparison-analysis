import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt # Visualize data
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # OMP: Error #15 해결


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")    

EPOCHS = 300
BATCH_SIZE = 128

train_loader = torch.utils.data.DataLoader(datasets.CIFAR100('./.data', train=True, download=True,
                transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])), batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader( datasets.CIFAR100('./.data', train=False, 
                transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])), batch_size=BATCH_SIZE, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=5)    #feature map 1->50
        self.conv2 = nn.Conv2d(50, 100, kernel_size=5)  #feature map 5->100
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2500, 500) # 2500 -> 500
        self.fc2 = nn.Linear(500, 100) # 500 -> 100

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 2500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
model = CNN().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    if batch_idx % 200 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    return loss
        
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

train_loss_list = []
test_loss_list = []

for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader, optimizer,epoch)
    train_loss_list.append(train_loss.cpu())
    test_loss, test_accuracy = evaluate(model, test_loader)
    test_loss_list.append(test_loss)
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))
torch.save(model.state_dict(), 'CNN_CIFAR100/cnn_model.pt')

# plot loss
train_loss_list = [loss.detach() for loss in train_loss_list]
plt.plot(range(1,EPOCHS+1), np.array(train_loss_list),'r', label = 'train_loss')
plt.plot(range(1,EPOCHS+1), np.array(test_loss_list),'b', label = 'test_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()
