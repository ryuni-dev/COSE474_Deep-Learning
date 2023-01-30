import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time

########################################
# You can define whatever classes if needed
########################################

class Block1(nn.Module):
    def __init__(self, size) -> None:
        super(Block1, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(size)
        self.bn2 = nn.BatchNorm2d(size*2)
        self.conv1 = nn.Conv2d(in_channels=size, out_channels=size*2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=size*2, out_channels=size*2, kernel_size=3, stride=1, padding=1)
        self.skip_connection = nn.Conv2d(in_channels=size, out_channels=size*2, kernel_size=1, stride=2)
        
    def forward(self, x):
        bn1 = self.bn1(x)
        relu1 = F.relu(bn1)

        skip_connection = self.skip_connection(relu1)

        conv1 = self.conv1(relu1)
        bn2 = self.bn2(conv1)
        relu2 = F.relu(bn2)
        conv2 = self.conv2(relu2)

        out = conv2 + skip_connection

        return out

class Block2(nn.Module):
    def __init__(self, size) -> None:
        super(Block2, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(size)
        self.bn2 = nn.BatchNorm2d(size)
        self.conv1 = nn.Conv2d(in_channels=size, out_channels=size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=size, out_channels=size, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bn1 = self.bn1(x)
        relu1 = F.relu(bn1)
        conv1 = self.conv1(relu1)
        bn2 = self.bn2(conv1)
        relu2 = F.relu(bn2)
        conv2 = self.conv2(relu2)

        out = conv2 + x
        
        return out




class IdentityResNet(nn.Module):
    
    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()
    ########################################
    # Implement the network
    # You can declare whatever variables
    ########################################
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.stage1 = nn.Sequential(
            *([Block2(64)] * nblk_stage1)
        )
        self.stage2 = nn.Sequential(
            *([Block1(64)] + [Block2(128)] * (nblk_stage2-1))
        )
        self.stage3 = nn.Sequential(
            *([Block1(128)] + [Block2(256)] * (nblk_stage3-1))
        )
        self.stage4 = nn.Sequential(
            *([Block1(256)] + [Block2(512)] * (nblk_stage4-1))
        )

        self.fc = nn.Linear(in_features=512, out_features=10)

    ########################################
    # You can define whatever methods
    ########################################

    def forward(self, x):
        ########################################
        # Implement the network
        # You can declare or define whatever variables or methods
        ########################################
        conv = self.conv(x)
        stage1 = self.stage1(conv)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        pl = F.avg_pool2d(stage4, kernel_size=4, stride=4)
        out = pl.squeeze()
        out = self.fc(out)

        return out

########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('current device: ', dev)


########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 32

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
net.to(dev)


# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
optimizer = optim.SGD(net.parameters(), lr=0.007, momentum=0.9)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)
        
        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        optimizer.zero_grad()
        
        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = net(inputs)
        
        # set loss
        loss = criterion(outputs, labels)
        
        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()
        
        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end-t_start, ' sec')
            t_start = t_end

print('Finished Training')


# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' %(classes[i]), ': ',
          100 * class_correct[i] / class_total[i],'%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct)/sum(class_total))*100, '%')


