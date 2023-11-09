import os
import struct
import socket
import pickle
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model_client import ResNet50 as ResNet50_client
from model_server import ResNet50 as ResNet50_server
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from utils import get_logger, get_metrics
from torch.utils.data import Subset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--connection_start_from_client', action='store_true', default=False)
parser.add_argument('--client_in_sambanova', action='store_true', default=False)
args = parser.parse_args()


logger, exp_seq = get_logger(filename_prefix="client_")
logger.info(f"-------------------------Session: Exp {exp_seq}")
root_path = '../models/cifar10_data'

# Setup cpu
device = 'cpu'
# device = 'cuda:0'
torch.manual_seed(777)

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
    global total_communication_time_server_to_client
    global offset_time
    total_communication_time_server_to_client += time.time() - msg['communication_time_stamp'] + offset_time
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



lr = 0.001
resnet_client = ResNet50_client(channel=3).to(device) # parameters depend on the dataset
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_client.parameters(), lr = lr, momentum = 0.9)

# Training 



total_communication_time_server_to_client = 0
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
    offset_time = - total_communication_time_server_to_client
    total_communication_time_server_to_client = 0
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
    offset_time = - total_communication_time_server_to_client
    total_communication_time_server_to_client = 0
    s1 = conn
    send_msg(s1, {"initial_msg": "Greetings from client"})


start_time = time.time()
total_validation_time = 0
epoch_communication_time_server_to_client = 0
epoch_communication_time_client_to_server = 0
epoch_received_msg_len = 0
training_time = 0
# training_time_server = 0
epoch = args.epoch
msg = {
    'epoch': epoch,
    'total_batch': total_batch,
    'lr': lr
}
send_msg(s1, msg) # send 'epoch' and 'batch size' to server




for epc in range(epoch):
    epoch_start_time = time.time()
    epoch_training_time = 0
    epoch_training_time_server = 0
    
    # logger.info(f"running epoch  {epc+1}")
    resnet_client.train()
    target = 0

    for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Training with {}'.format(remote_server))):
        batch_training_start_time = time.time()
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
        epoch_training_time += time.time() - batch_training_start_time
        
        
        send_msg(s1, msg) # send label and output(feature) to server
        # batch_training_start_time_server = time.time()
        rmsg = recv_msg(s1) # receive gradaint after the server has completed the back propagation.
        # epoch_training_time_server += time.time() - batch_training_start_time_server

        batch_training_start_time = time.time()
        client_grad = rmsg['grad']
        output.backward(client_grad) # continue back propagation for client side layers.
        optimizer.step()
        epoch_training_time += time.time() - batch_training_start_time

        if(i+1) % 100 == 0:
            pass
            # break

    training_time += epoch_training_time 
    # training_time_server += epoch_training_time_server
    
    # validation after each epoch
    server_model_size = received_msg_len
    rmsg = recv_msg(s1)
    server_model_size = received_msg_len - server_model_size
    server_model_state_dict = rmsg['server model']
    resnet_server = ResNet50_server(num_classes=10).to(device)
    resnet_server.load_state_dict(server_model_state_dict)

    validation_start_time = time.time()
    # train_loss, train_acc, train_auc, train_bal_acc = get_metrics(resnet_server, resnet_client, train_loader, criterion, device)
    test_loss, test_acc, test_auc, test_bal_acc = get_metrics(resnet_server, resnet_client, test_loader, criterion, device)
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
    send_msg(s1, {'server_to_client_communication_time': total_communication_time_server_to_client})
    rmsg = recv_msg(s1)
    total_communication_time_client_to_server = rmsg['client_to_server_communication_time']
    logger.info(f"Epoch: client to server com time: {round(total_communication_time_client_to_server - epoch_communication_time_client_to_server, 2)}")
    logger.info(f"Epoch: server to client com. time: {round(total_communication_time_server_to_client - epoch_communication_time_server_to_client, 2)}")
    epoch_communication_time_server_to_client = total_communication_time_server_to_client    
    epoch_communication_time_client_to_server = total_communication_time_client_to_server
    logger.info(f"Epoch: training time client: {round(epoch_training_time, 2)}")
    # logger.info(f"Epoch: training time server (over-approximation): {round(epoch_training_time_server, 2)}")
    logger.info(f"Epoch: validation time: {round(msg['validation time'], 2)}")
    logger.info(f"Epoch: total time: {round(time.time() - epoch_start_time, 2)}")
    logger.info(f"Epoch: received msg len from server: {round((received_msg_len - epoch_received_msg_len)/1024/1024, 2)} MB")
    epoch_received_msg_len = received_msg_len
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
logger.info(f"Training time client: {round(training_time, 2)}")
# logger.info(f"Training time server (over-approximation): {round(training_time_server, 2)}")
logger.info(f"Validation time: {round(total_validation_time, 2)}")
logger.info(f"Total time: {round(end_time - start_time, 2)}")
logger.info(f"Received msg len from server: {round(received_msg_len/1024/1024, 2)} MB")
# 