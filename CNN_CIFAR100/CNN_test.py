import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")    

BATCH_SIZE = 128

test_loader = torch.utils.data.DataLoader( datasets.CIFAR100('./.data', train=False, download = True,
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
    
model = CNN()
model.load_state_dict(torch.load('20182106/CNN_CIFAR100/cnn_model.pt'))
model.to(DEVICE)

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

test_loss, test_accuracy = evaluate(model, test_loader)
print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
