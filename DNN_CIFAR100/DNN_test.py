import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

BATCH_SIZE = 128

transform = transforms.Compose([
transforms.ToTensor()
])

testset = datasets.CIFAR100(
    root = './.data/', 
    train = False,
    download = True,
    transform = transform
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
    
model = Net()
model.load_state_dict(torch.load('DNN_CIFAR100/dnn_model.pt'))
model.to(DEVICE)

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


test_loss, test_accuracy = evaluate(model, test_loader)
print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
