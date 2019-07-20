# 导入必要的库
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans
import time

# 超参数
BATCH_SIZE = 100
# 损失函数
loss_func = nn.CrossEntropyLoss()
# 可以在CPU或者GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR10的输入图片各channel的均值和标准差
mean = [x/255 for x in [125.3, 23.0, 113.9]]
std = [x/255 for x in [63.0, 62.1, 66.7]]
n_train_samples = 50000


# 多进程需要加一个main函数，否则会报错
if __name__ == '__main__':
    # 数据增强-->训练集
    train_set = dsets.CIFAR10(root='./数据集/CIFAR10/',
                              train=True,
                              download=False,
                              transform=trans.Compose([
                                 trans.RandomHorizontalFlip(),
                                 trans.RandomCrop(32, padding=4),
                                 trans.ToTensor(),
                                 trans.Normalize(mean, std)
                             ]))
    train_dl = DataLoader(train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)     # 如需多线程，可以自行更改
    # train_set.train_data = train_set.train_data[0:n_train_samples]
    # train_set.train_labels = train_set.train_labels[0:n_train_samples]

    #  -->测试集
    test_set = dsets.CIFAR10(root='./数据集/CIFAR10/',
                             train=False,
                             download=False,
                             transform=trans.Compose([
                                trans.ToTensor(),
                                trans.Normalize(mean, std)
                            ]))

    test_dl = DataLoader(test_set,
                         batch_size=BATCH_SIZE,
                         num_workers=0)      # 如需多线程，可以自行更改
# train_set.train_data = train_set.train_data[:n_train_samples]
# train_set.train_labels = train_set.train_labels[:n_train_samples]

# 定义卷积层
def conv3x3(in_features, out_features):
    return nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)


# VGG19
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 1
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 2
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 3
            conv3x3(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 4
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 5
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 6
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 8
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 9
            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 10
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 11
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 12
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 13
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 15
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 16
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )

        self.classifier = nn.Sequential(
            # 17
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 18
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 19
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# 定义训练的辅助函数 包含error与accuracy
def eval(model, loss_func, dataloader):

    model.eval()
    loss, accuracy = 0, 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            error = loss_func(logits, batch_y)
            loss += error.item()

            probs, pred_y = logits.data.max(dim=1)
            accuracy += (pred_y==batch_y.data).float().sum()/batch_y.size(0)

    loss /= len(dataloader)
    accuracy = accuracy*100.0/len(dataloader)
    return loss, accuracy


def train_epoch(model, loss_func, optimizer, dataloader):

    model.train()
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        error = loss_func(logits, batch_y)
        error.backward()
        optimizer.step()


nepochs = 50
vgg19 = VGG().to(device)
# 可以尝试打印出vgg19的网络结构
# print(vgg19)

optimizer = torch.optim.SGD(vgg19.parameters(), lr=0.01, momentum=0.9, nesterov=True)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
learn_history = []

print('开始训练VGG19……')

for epoch in range(nepochs):
    since = time.time()
    train_epoch(vgg19, loss_func, optimizer, train_dl)

    if (epoch)%5 == 0:
        tr_loss, tr_acc = eval(vgg19, loss_func, train_dl)
        te_loss, te_acc = eval(vgg19, loss_func, test_dl)
        learn_history.append((tr_loss, tr_acc, te_loss, te_acc))
        now = time.time()
        print('[%3d/%d, %.0f seconds]|\t tr_err: %.1e, tr_acc: %.2f\t |\t te_err: %.1e, te_acc: %.2f'%(
            epoch+1, nepochs, now-since, tr_loss, tr_acc, te_loss, te_acc))
