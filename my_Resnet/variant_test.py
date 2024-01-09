import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

# CUDA 사용 가능 여부 확인 및 디바이스 설정
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

BATCH_SIZE = 128

# 테스트 데이터 로더 설정
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./.data',
        train=False,
        download = True,
        transform=transforms.Compose([
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 정규화
    )),
    batch_size=BATCH_SIZE, shuffle=True)  # 미니배치 크기와 데이터 섞기

# ResNet의 기본 블록 정의
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 첫 번째 컨볼루션 레이어
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 첫 번째 배치 정규화 레이어
        # 두 번째 컨볼루션 레이어
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 두 번째 배치 정규화 레이어
        self.shortcut = nn.Sequential()
        # 스트라이드가 1이 아니거나 입력 레이어의 채널 수가 출력 레이어의 채널 수와 다를 경우
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),  # 1x1 컨볼루션
                nn.BatchNorm2d(planes)  # 배치 정규화
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 첫 번째 컨볼루션, ReLU 활성화
        out = self.bn2(self.conv2(out))  # 두 번째 컨볼루션, 배치 정규화
        out += self.shortcut(x)  # Residual 연결
        out = F.relu(out)  # ReLU 활성화
        return out

# 사용자 정의 ResNet 모델 정의
class my_ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(my_ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 18, stride=1)  # 첫 번째 레이어에 18개의 블록
        self.layer2 = self._make_layer(32, 18, stride=2)  # 두 번째 레이어에 18개의 블록
        self.layer3 = self._make_layer(64, 18, stride=2)  # 세 번째 레이어에 18개의 블록
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        # 스트라이드 값 계산
        strides = [stride] + [1] * (num_blocks - 1)
        # BasicBlock을 담을 빈 리스트 생성
        layers = []
        for stride in strides:
            # BasicBlock을 생성하고 리스트에 추가
            layers.append(BasicBlock(self.in_planes, planes, stride))
            # in_planes 값을 현재 planes 값으로 업데이트
            self.in_planes = planes
        # 모든 레이어를 시퀀셜 레이어로 묶어 반환
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 입력 레이어: 컨볼루션, ReLU 활성화
        out = self.layer1(out)  # 첫 번째 레이어
        out = self.layer2(out)  # 두 번째 레이어
        out = self.layer3(out)  # 세 번째 레이어
        out = F.avg_pool2d(out, 8)  # 8x8 평균 풀링
        out = out.view(out.size(0), -1)  # 1차원으로 평탄화
        out = self.linear(out)  # 선형 레이어 (분류)
        return out

# 학습된 모델 불러오고 디바이스 설정
model = my_ResNet()
model.load_state_dict(torch.load('my_Resnet/res_model.pt'))
model.to(DEVICE)

# 평가 함수 정의
def evaluate(model, test_loader):
    model.eval()  # 모델을 평가 모드로 설정
    test_loss = 0  # 테스트 손실 초기화
    correct = 0  # 정확한 예측 수 초기화
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)  # 테스트 데이터를 지정한 장치로 이동
            output = model(data)  # 모델에 입력 데이터를 전달하여 예측 얻음
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 손실 누적
            pred = output.max(1, keepdim=True)[1]  # 예측 중 가장 높은 확률의 클래스 선택
            correct += pred.eq(target.view_as(pred)).sum().item()  # 올바른 예측 수 누적
    test_loss /= len(test_loader.dataset)  # 테스트 손실을 데이터셋 크기로 나누어 평균 계산
    test_accuracy = 100. * correct / len(test_loader.dataset)  # 정확도 계산
    return test_loss, test_accuracy  # 평균 테스트 손실과 정확도 반환

# 모델을 사용하여 테스트 데이터 평가
test_loss, test_accuracy = evaluate(model, test_loader)
print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
