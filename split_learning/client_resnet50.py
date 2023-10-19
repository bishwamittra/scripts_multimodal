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
from utils import get_logger
from torch.utils.data import Subset

import argparse
parser = argparse.ArgumentParser()
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





# Training hyper parameters
# 


resnet_client = ResNet50_client(channel=3).to(device) # parameters depend on the dataset

lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_client.parameters(), lr = lr, momentum = 0.9)

# Training 



total_communication_time = 0
offset_time = 0
if(not args.connection_start_from_client):
    # host = '10.2.144.188'
    # host = '10.9.240.14'
    host = '10.2.143.109'
    port = 10081
    s1 = socket.socket()
    s1.connect((host, port)) # establish connection
    send_msg(s1, {"initial_msg": "Greetings from client"}) # send 'epoch' and 'batch size' to server
    remote_server = recv_msg(s1)['server_name'] # get server's meta information.
    offset_time = - total_communication_time
    total_communication_time = 0
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
    offset_time = - total_communication_time
    total_communication_time = 0
    s1 = conn
    send_msg(s1, {"initial_msg": "Greetings from client"})


start_time = time.time()
epoch = 1
msg = {
    'epoch': epoch,
    'total_batch': total_batch
}
send_msg(s1, msg) # send 'epoch' and 'batch size' to server




for epc in range(epoch):
    logger.info(f"running epoch  {epc}")

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

        if(i+1) % 10 == 0:
            logger.info(f"Server to client communication time: {round(total_communication_time, 2)}")
            send_msg(s1, {'server_to_client_communication_time': round(total_communication_time, 2)})

            break       
        

    # validation after each epoch      
    rmsg = recv_msg(s1)
    server_model_state_dict = rmsg['server model']
    resnet_server = ResNet50_server(num_classes=10).to(device)
    resnet_server.load_state_dict(server_model_state_dict)
    resnet_server.eval()
    resnet_client.eval()
    test_dataset_size = len(test_loader.dataset)
    with torch.no_grad():
        logits_all, targets_all = torch.tensor([], device=device), torch.tensor([], dtype=torch.int, device=device)
        for x, label in tqdm(test_loader):
            x = x.to(device)
            label = label.to(device)
            output = resnet_client(x)
            client_output = output.clone().detach().requires_grad_(True)

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

        send_msg(s1, {
            'Test Loss': round(test_loss.item(), 2),
            'Test Accuracy': round(test_acc, 2),
            'Test AUC': round(test_auc, 2),
            'Test Balanced Accuracy': round(test_bal_acc, 2)
        })



send_msg(s1, {'server_to_client_communication_time': total_communication_time})       
     
s1.close()

end_time = time.time()

# 