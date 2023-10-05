# %%
server_name = 'SERVER_001'

import os
import h5py

import socket
import struct
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import time
import sys
from utils import get_logger



from tqdm import tqdm


from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable
import torch.nn.init as init
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import copy

# %%
# Setup CUDA
seed_num = 777
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch.manual_seed(seed_num)
# if device == "cuda:0":
#     torch.cuda.manual_seed_all(seed_num)
device = "cpu"
logger, exp_seq = get_logger()
logger.info(f"-------------------------Session: Exp {exp_seq}")

# %%
def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    l_send = len(msg)
    msg = struct.pack('>I', l_send) + msg
    sock.sendall(msg)
    return l_send

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msg_len = recv_all(sock, 4)
    if not raw_msg_len:
        return None
    msg_len = struct.unpack('>I', raw_msg_len)[0]
    # read the message data
    msg =  recv_all(sock, msg_len)
    msg = pickle.loads(msg)
    return msg, msg_len

def recv_all(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# %% [markdown]
# The model definition of server

# %%
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
    

class Bottleneck_server(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_server, self).__init__()
        self.norm = norm
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
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
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    # def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
    def __init__(self, block, block_server, num_blocks, num_classes=10, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        # self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer1 = self._make_layer_server(block, block_server, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _make_layer_server(self, block, block_server, planes, num_blocks, stride):
        # strides = [stride] + [1]*(num_blocks-1)
        layers = []
        layers.append(block_server(self.in_planes, planes, 1))
        self.in_planes = planes * block.expansion
        layers.append(block(self.in_planes, planes, 1))
        self.in_planes = planes * block.expansion
        layers.append(block(self.in_planes, planes, 1))
        self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
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



def ResNet50(num_classes):
    return ResNet(Bottleneck, Bottleneck_server, [3,4,6,3], num_classes=num_classes)

# %%
resnet_server =  ResNet50(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = optim.SGD(resnet_server.parameters(), lr=lr, momentum=0.9)

epochs = 1

# %%
resnet_server

# %%
# host = '10.2.144.188'
# host = '10.9.240.14'
host = '10.2.143.109'
port = 10081

s = socket.socket()
s.bind((host, port))
s.listen(5)

conn, addr = s.accept()
logger.info(f"Connected to: {addr}")

# read epoch
rmsg, data_size = recv_msg(conn) # receive total bach number and epoch from client.

epoch = rmsg['epoch']
num_batch = rmsg['total_batch']

logger.info(f"received epoch: {rmsg['epoch']}, {rmsg['total_batch']}")

send_msg(conn, server_name) # send server meta information.

# Start training
start_time = time.time()
logger.info(f"Start training @ {time.asctime()}")

for epc in range(epoch):
    init = 0
    for i in tqdm(range(num_batch), ncols = 100, desc='Training with {}'.format(server_name)):
        optimizer.zero_grad()
        
        msg, data_size = recv_msg(conn) # receives label and feature from client.
        
        # label
        label = msg['label']
        label = label.clone().detach().long().to(device) # conversion between gpu and cpu.
        
        # feature
        client_output_cpu = msg['client_output']
        client_output = client_output_cpu.to(device)

        # forward propagation
        output = resnet_server(client_output)
        loss = criterion(output, label) # compute cross-entropy loss
        loss.backward() # backward propagation
        
        # send gradient to client
        msg = client_output_cpu.grad.clone().detach()
        data_size = send_msg(conn, msg)
        
        optimizer.step()
        

        if (i + 1) % 100 == 0:

            # measure accuracy and record loss
            _, predicted = torch.max(output, 1)
            correct = (predicted == label).sum().item()
            accuracy = correct / len(label)
            logger.info(f'Epoch: {epc+1}/{epoch}, Batch: {i+1}/{num_batch}, Train Loss: {round(loss.item(), 2)} Train Accuracy: {round(accuracy, 2)}')

        if (i + 1) % 1000 == 0:
            logger.info("Start validation")
            # validation
            rmsg, data_size = recv_msg(conn) # receive total bach number and epoch from client.
            num_test_batch = rmsg['num_batch']
            test_dataset_size = rmsg['dataset_size']
            resnet_server.eval()
            with torch.no_grad():
                logits_all, targets_all = torch.tensor([], device='cpu'), torch.tensor([], dtype=torch.int, device='cpu')
                # for j in range(num_test_batch):
                for j in range(num_test_batch):
                    msg, data_size = recv_msg(conn)
                    # label
                    label = msg['label']
                    label = label.clone().detach().long().to(device) # conversion between gpu and cpu.

                    # feature
                    client_output_cpu = msg['client_output']
                    client_output = client_output_cpu.to(device)
                    
                    # forward propagation
                    logits = resnet_server(client_output)
                    logits_all = torch.cat((logits_all, logits.detach().cpu()),dim=0)
                    targets_all = torch.cat((targets_all, label.cpu()), dim=0)

                pred = F.log_softmax(logits_all, dim=1)
                test_loss = criterion(pred, targets_all)/test_dataset_size # validation loss
                
                output = pred.argmax(dim=1) # predicated/output label
                prob = F.softmax(logits_all, dim=1) # probabilities

                test_acc = accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
                test_bal_acc = balanced_accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
                test_auc = roc_auc_score(targets_all.numpy(), prob.numpy(), multi_class='ovr')
                logger.info(f'Test Loss: {round(test_loss.item(), 2)} Test Accuracy: {round(test_acc, 2)} Test AUC: {round(test_auc, 2)} Test Balanced Accuracy: {round(test_bal_acc, 2)}')

            # break

logger.info(f'Contribution from {server_name} is done')
logger.info(f'Contribution duration is: {time.time() - start_time} seconds')

# %%



