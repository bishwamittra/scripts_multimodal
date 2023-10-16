
import os
import struct
import socket
import pickle
import time

import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import Subset
from torch.autograd import Variable
import torch.nn.init as init
import copy



root_path = '../models/cifar10_data'

# Setup cpu
device = 'cpu'
# device = 'cuda:0'
torch.manual_seed(777)

# Setup client order
client_order = int(0)
print('Client starts from: ', client_order)

num_train_data = 50000

# Load data
from random import shuffle

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

indices = list(range(50000))

part_tr = indices[num_train_data * client_order : num_train_data * (client_order + 1)]

train_set  = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform)
train_set_sub = Subset(train_set, part_tr)
train_loader = torch.utils.data.DataLoader(train_set_sub, batch_size=8, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False, num_workers=2)

x_train, y_train = next(iter(train_loader))
print(f'Train batch shape x: {x_train.size()} y: {y_train.size()}')
total_batch = len(train_loader)
print(f'Num Batch {total_batch}')



# Helper functions for communication between client and server.
def send_msg(sock, msg):
    assert isinstance(msg, dict)
    msg['communication_time_stamp'] = time.time()
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msg_len = recv_all(sock, 4)
    if not raw_msg_len:
        return None
    msg_len = struct.unpack('>I', raw_msg_len)[0]
    # read the message data
    msg =  recv_all(sock, msg_len)
    msg = pickle.loads(msg)
    global total_communication_time
    global offset_time
    total_communication_time += time.time() - msg['communication_time_stamp'] + offset_time
    return msg

def recv_all(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# Definition of client side model (input layer only)


''' ResNet '''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    # def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
    def __init__(self, channel=3, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.classifier = nn.Linear(512*block.expansion, num_classes)

    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride, self.norm))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return out

    # def embed(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     out = out.view(out.size(0), -1)
    #     return out

def ResNet50(channel):
    return ResNet(channel=channel)



# Training hyper parameters
# 


resnet_client = ResNet50(channel=3).to(device) # parameters depend on the dataset

lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_client.parameters(), lr = lr, momentum = 0.9)

# Training 



# host = '10.2.144.188'
# host = '10.9.240.14'
host = '10.2.143.109'
port = 10081
epoch = 10

start_time = time.time()

s1 = socket.socket()
s1.connect((host, port)) # establish connection
# s1.close()


msg = {
    'epoch': epoch,
    'total_batch': total_batch
}

send_msg(s1, msg) # send 'epoch' and 'batch size' to server

# resnet_client.eval() # Why eval()?
total_communication_time = 0
offset_time = 0
remote_server = recv_msg(s1)['server_name'] # get server's meta information.
offset_time = - total_communication_time



for epc in range(epoch):
    print("running epoch ", epc)

    target = 0

    for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Training with {}'.format(remote_server))):
        x, label = data
        x = x.to(device)
        label = label.to(device)
        optimizer.zero_grad()


        output = resnet_client(x)
        client_output = output.clone().detach().requires_grad_(True)

        msg = {
            'label': label,
            'client_output': client_output
        }
        
        send_msg(s1, msg) # send label and output(feature) to server
        rmsg = recv_msg(s1) # receive gradaint after the server has completed the back propagation.
        client_grad = rmsg['grad']
        
        

        output.backward(client_grad) # continue back propagation for client side layers.
        optimizer.step()


        if (i + 1) % 1000 == 0:
            msg = {
                'num_batch': len(test_loader),
                'dataset_size': len(test_loader.dataset)
            }
            # print(msg)
            send_msg(s1, msg) # 'num test batch' to server
            resnet_client.eval()
            with torch.no_grad():
                for x, label in test_loader:
                    x = x.to(device)
                    label = label.to(device)
                    output = resnet_client(x)
                    client_output = output.clone().detach().requires_grad_(True)
                    msg = {
                        'label': label,
                        'client_output': client_output
                    }
                    send_msg(s1, msg) # send label and output(feature) to server

            # break
            send_msg(s1, {'server_to_client_communication_time': round(total_communication_time, 2)})       
            # break
        
        if(i+1) % 100 == 0:
            print(f"Server to client communication time: {round(total_communication_time, 2)}")
            


send_msg(s1, {'server_to_client_communication_time': total_communication_time})       
     
s1.close()

end_time = time.time()

# 





