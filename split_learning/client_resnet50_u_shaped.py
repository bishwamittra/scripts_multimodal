import os
import struct
import socket
import pickle
from itertools import chain
import time
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model_client_u_shaped import ResNet50_Head as ResNet50_client_head
from model_client_u_shaped import ResNet50_Tail as ResNet50_client_tail
from model_server_u_shaped import ResNet50 as ResNet50_server
from utils import get_logger, get_metrics_u_shaped
from torch.utils.data import Subset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int)
parser.add_argument('--connection_start_from_client', action='store_true', default=False)
parser.add_argument('--client_in_sambanova', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()


logger, exp_seq = get_logger(filename_prefix="client_")
logger.info(f"-------------------------Session: Exp {exp_seq}")
root_path = '../models/cifar10_data'

# Setup cpu
device = 'cpu'
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)     


# Setup client order
client_order = int(0)
logger.info(f'Client starts from: {client_order}')

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
train_loader = torch.utils.data.DataLoader(train_set_sub, batch_size=64, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

x_train, y_train = next(iter(train_loader))
logger.info(f'Train batch shape x: {x_train.size()} y: {y_train.size()}')
total_batch = len(train_loader)
logger.info(f'Num Batch {total_batch}')



def sync_time(conn, logger):
    # a back and forth communication to sync time between client and server.

    global epoch_communication_time_server_to_client
    global offset_time
    epoch_communication_time_server_to_client = 0
    offset_time = 0
    rmsg = recv_msg(conn)
    logger.info(rmsg['sync_time'])
    send_msg(conn, {"sync_time": "sync request from client"})
    
    
    offset_time = - epoch_communication_time_server_to_client
    epoch_communication_time_server_to_client = 0
    


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
    global received_msg_len
    received_msg_len += msg_len
    # read the message data
    msg =  recv_all(sock, msg_len)
    msg = pickle.loads(msg)
    global epoch_communication_time_server_to_client
    global offset_time
    epoch_communication_time_server_to_client += time.time() - msg['communication_time_stamp'] + offset_time
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





# Training hyper parameters
# 


resnet_client_head = ResNet50_client_head(channel=3).to(device) # parameters depend on the dataset
resnet_client_tail = ResNet50_client_tail(num_classes=10).to(device) # parameters depend on the dataset

lr = 0.001
criterion = nn.CrossEntropyLoss()
# all_params = chain(resnet_client_head.parameters(), resnet_client_tail.parameters())
# optimizer_combined = optim.SGD(all_params, lr = lr, momentum = 0.9)
optimizer_tail = optim.SGD(resnet_client_tail.parameters(), lr = lr, momentum = 0.9)
optimizer_head = optim.SGD(resnet_client_head.parameters(), lr = lr, momentum = 0.9)

# Training 



epoch_communication_time_server_to_client = 0
offset_time = 0
received_msg_len = 0
if(not args.connection_start_from_client):
    # host = '10.2.144.188'
    # host = '10.9.240.14'
    host = '10.2.143.109'
    port = 10081
    s1 = socket.socket()
    s1.connect((host, port)) # establish connection
    send_msg(s1, {"initial_msg": "Greetings from client"}) # send 'epoch' and 'batch size' to server
    remote_server = recv_msg(s1)['server_name'] # get server's meta information.
    logger.info(f"Server: {remote_server}")

else:
    if(args.client_in_sambanova):
        host = '10.9.240.14'
        port = 8870
    else:
        host = '10.2.143.109'
        port = 10081

    s1 = socket.socket()
    s1.bind((host, port))
    s1.listen(5)
    conn, addr = s1.accept()
    logger.info(f"Connected to: {addr}")
    rmsg = recv_msg(conn) 
    print(rmsg['initial_msg'])
    remote_server = rmsg['server_name']
    s1 = conn
    send_msg(s1, {"initial_msg": "Greetings from client"})



start_time = time.time()
total_validation_time = 0
total_training_time = 0
total_communication_time_server_to_client = 0
total_communication_time_client_to_server = 0
epoch_received_msg_len = 0
total_size_server_model = 0
total_size_server_output = 0
total_size_client_head_gradient = 0



# training_time_server = 0
epoch = args.epoch
msg = {
    'epoch': epoch,
    'total_batch': total_batch
}
send_msg(s1, msg) # send 'epoch' and 'batch size' to server

"""
Todo: 
    1. Version 1: Separate optimizers for head and tail. Once tail computes gradient, we call optimizer_tail.step(). Similarly for head
    2. Version 2: One optmizer combining both parameters.
    3. Find the model definition of VIT
"""


for epc in range(epoch):
    sync_time(s1, logger)
    epoch_start_time = time.time()
    epoch_training_time = 0
    epoch_training_time_server = 0
    epoch_size_server_output = 0
    epoch_size_client_head_gradient = 0
    epoch_communication_time_server_to_client = 0
    
    # logger.info(f"running epoch  {epc+1}")
    resnet_client_head.train()
    resnet_client_tail.train()
    target = 0

    for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Training with {}'.format(remote_server))):
        
        # head forward propagation
        batch_training_start_time = time.time()
        x, label = data
        x = x.to(device)
        label = label.to(device)
        optimizer_head.zero_grad()
        optimizer_tail.zero_grad()
        output_head = resnet_client_head(x)
        client_output_head = output_head.clone().detach().requires_grad_(True)
        msg = {
            'client_output': client_output_head
        }
        epoch_training_time += time.time() - batch_training_start_time
        
        
        send_msg(s1, msg) # send head output to server
        
        
        size_server_output = received_msg_len
        rmsg = recv_msg(s1) # receive server output from server
        size_server_output = received_msg_len - size_server_output
        epoch_size_server_output += size_server_output


        # tail forward propagation
        batch_training_start_time = time.time()
        server_output_gpu = rmsg['server_output']
        server_output = server_output_gpu.to(device)
        output_tail = resnet_client_tail(server_output)
        
        
        # backward propagation
        loss = criterion(output_tail, label) # compute cross-entropy loss
        loss.backward() # backward propagation
        epoch_training_time += time.time() - batch_training_start_time

        # send gradient to server
        msg = {
            "server_grad": server_output_gpu.grad.clone().detach(),
        }
        send_msg(s1, msg)


        optimizer_tail.step()

        size_client_head_gradient = received_msg_len
        rmsg = recv_msg(s1) # receive gradient of the head from server
        size_client_head_gradient = received_msg_len - size_client_head_gradient
        epoch_size_client_head_gradient += size_client_head_gradient

        # update head gradient
        batch_training_start_time = time.time()
        client_output_head.backward(rmsg['client_grad'])
        optimizer_head.step()
        epoch_training_time += time.time() - batch_training_start_time


        if(i+1) % 100 == 0:
            pass
            # break

    total_training_time += epoch_training_time 
    # training_time_server += epoch_training_time_server
    
    # validation after each epoch
    server_model_size = received_msg_len
    rmsg = recv_msg(s1)
    server_model_size = received_msg_len - server_model_size
    total_size_server_model += server_model_size
    server_model_state_dict = rmsg['server model']
    resnet_server = ResNet50_server().to(device)
    resnet_server.load_state_dict(server_model_state_dict)

    validation_start_time = time.time()
    # train_loss, train_acc, train_auc, train_bal_acc = get_metrics(resnet_server, resnet_client, train_loader, criterion, device)
    test_loss, test_acc, test_auc, test_bal_acc = get_metrics_u_shaped(resnet_server, resnet_client_head, resnet_client_tail, test_loader, criterion, device)
    msg = {
        # 'Train Loss': train_loss,
        # 'Train Accuracy': train_acc,
        # 'Train AUC': train_auc,
        # 'Train Balanced Accuracy': train_bal_acc,


        'Test Loss': test_loss,
        'Test Accuracy': test_acc,
        'Test AUC': test_auc,
        'Test Balanced Accuracy': test_bal_acc,
        'validation time': time.time() - validation_start_time
    }
    total_validation_time += msg['validation time']


    send_msg(s1, msg)
    logger.info("")
    logger.info(f"Epoch {epc+1}/{epoch} results:")
    # logger.info(f"Train Loss: {round(train_loss, 4)}, Train Accuracy: {round(train_acc, 4)}, Train AUC: {round(train_auc, 4)}, Train Balanced Accuracy: {round(train_bal_acc, 4)}")
    logger.info(f"Test Loss: {round(test_loss, 4)}, Test Accuracy: {round(test_acc, 4)}, Test AUC: {round(test_auc, 4)}, Test Balanced Accuracy: {round(test_bal_acc, 4)}")
    # communicating time
    send_msg(s1, {'server_to_client_communication_time': epoch_communication_time_server_to_client})
    rmsg = recv_msg(s1)
    epoch_communication_time_client_to_server = rmsg['client_to_server_communication_time']
    logger.info(f"Epoch: client to server com time: {round(epoch_communication_time_client_to_server, 2)}")
    logger.info(f"Epoch: server to client com. time: {round(epoch_communication_time_server_to_client, 2)}")
    total_communication_time_server_to_client += epoch_communication_time_server_to_client    
    total_communication_time_client_to_server += epoch_communication_time_client_to_server
    logger.info(f"Epoch: training time client: {round(epoch_training_time, 2)}")
    # logger.info(f"Epoch: training time server (over-approximation): {round(epoch_training_time_server, 2)}")
    logger.info(f"Epoch: validation time: {round(msg['validation time'], 2)}")
    logger.info(f"Epoch: total time: {round(time.time() - epoch_start_time, 2)}")
    
    
    
    logger.info("")
    logger.info(f"Epoch: received msg len from server: {round((received_msg_len - epoch_received_msg_len)/1024/1024, 2)} MB")
    epoch_received_msg_len = received_msg_len
    logger.info(f"Epoch: size of server output: {round(epoch_size_server_output/1024/1024, 2)} MB")
    total_size_server_output += epoch_size_server_output
    logger.info(f"Epoch: size of client head gradient: {round(epoch_size_client_head_gradient/1024/1024, 2)} MB")
    total_size_client_head_gradient += epoch_size_client_head_gradient
    logger.info(f"Epoch: server model size: {round(server_model_size/1024/1024, 2)} MB")

    



send_msg(s1, {'server_to_client_communication_time': total_communication_time_server_to_client})       
rmsg = recv_msg(s1)
total_communication_time_client_to_server = rmsg['client_to_server_communication_time']     
s1.close()
end_time = time.time()


logger.info("")
logger.info(f'Summary')
logger.info(f"Client to server communication time: {round(total_communication_time_client_to_server, 2)}")
logger.info(f"Server to client communication time: {round(total_communication_time_server_to_client, 2)}")
logger.info(f"Training time client: {round(total_training_time, 2)}")
# logger.info(f"Training time server (over-approximation): {round(training_time_server, 2)}")
logger.info(f"Validation time: {round(total_validation_time, 2)}")
logger.info(f"Total time: {round(end_time - start_time, 2)}")
logger.info("")
logger.info(f"Received msg len from server: {round(received_msg_len/1024/1024, 2)} MB")
logger.info(f"Total size of server output: {round(total_size_server_output/1024/1024, 2)} MB")
logger.info(f"Total size of client head gradient: {round(total_size_client_head_gradient/1024/1024, 2)} MB")
logger.info(f"Total size of server model: {round(total_size_server_model/1024/1024, 2)} MB")
# 