# 필요한 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os

# OMP 에러 해결을 위한 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 학습을 위한 디바이스 설정 (CPU 또는 GPU)
USE_CUDA = torch.cuda.is_available()  # CUDA 사용 가능 여부 확인
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")  # CUDA를 사용할 수 있으면 GPU를, 아니면 CPU를 사용

# 학습 파라미터 설정
EPOCHS = 300  # 전체 학습 에포크 횟수
BATCH_SIZE = 128  # 각 미니배치의 크기

# 데이터 증강을 적용한 학습 및 테스트 데이터 로더 생성
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./.data',
        train=True,
        download=True,
        transform=transforms.Compose([
            # 데이터 증강을 위한 여러 변환을 적용
            transforms.RandomVerticalFlip(),  # 수직으로 뒤집어 augmentation 적용
            transforms.RandomRotation(degrees=(-100, 100)),  # -100도 ~ 100도 사이로 rotation을 적용한 augmentation
            transforms.RandomAffine(degrees=(0, 180), shear=20),  # affineTransform
            transforms.RandomCrop(32, padding=4),  # 무작위로 이미지를 자르는 것으로 데이터 다양성 증가
            transforms.RandomHorizontalFlip(),  # 수평으로 뒤집어 augmentation 적용
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 정규화
    )),
    batch_size=BATCH_SIZE, shuffle=True)  # 미니배치 크기와 데이터 섞기

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./.data',
        train=False,
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


# 모델, 옵티마이저, 학습률 스케줄러 설정
model = my_ResNet().to(DEVICE)  # 모델을 디바이스에 올림
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)  # SGD 옵티마이저 설정
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 학습률 스케줄러 설정

# 학습 함수 정의
def train(model, train_loader, optimizer):
    model.train()  # 모델을 학습 모드로 설정
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)  # 학습 데이터를 지정한 장치로 이동
        optimizer.zero_grad()  # 그래디언트 초기화
        output = model(data)  # 모델에 입력 데이터를 전달하여 예측 얻음
        loss = F.cross_entropy(output, target)  # 손실 계산
        loss.backward()  # 역전파 수행
        optimizer.step()  # 옵티마이저로 모델 파라미터 업데이트
    return loss  # 마지막 미니배치의 손실 반환


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


# 학습 및 테스트 손실을 저장하는 리스트
train_loss_list = []
test_loss_list = []

# 학습 루프
for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader, optimizer)  # 학습 함수를 호출하여 학습을 수행하고 학습 손실을 얻음
    train_loss_list.append(train_loss.cpu())  # 학습 손실을 리스트에 추가 (CPU로 변환)
    scheduler.step()  # 학습률 스케줄러를 업데이트
    test_loss, test_accuracy = evaluate(model, test_loader)  # 평가 함수를 호출하여 테스트 손실과 정확도를 얻음
    test_loss_list.append(test_loss)  # 테스트 손실을 리스트에 추가
    # 현재 에포크에서의 테스트 손실과 정확도를 출력
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))


# 학습된 모델 저장
torch.save(model.state_dict(), 'my_Resnet/res_model.pt')

# plot loss

# train_loss_list에 있는 각 손실 값들을 detach()를 사용하여 그래프 표시를 위해 분리
train_loss_list = [loss.detach() for loss in train_loss_list]
# 학습 및 테스트 손실 그래프 그리기
plt.plot(range(1, EPOCHS + 1), np.array(train_loss_list), 'r', label='train_loss')
plt.plot(range(1, EPOCHS + 1), np.array(test_loss_list), 'b', label='test_loss')
plt.xlabel('epoch')  # x축 레이블 설정
plt.ylabel('loss')  # y축 레이블 설정
plt.legend(loc='upper right')  # 그래프에 범례 추가
plt.show()  # 그래프 표시
