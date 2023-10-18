
server_name = 'SERVER_001'

import socket
import struct
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time
from model_server import ResNet50 as ResNet50_server
from utils import get_logger



from tqdm import tqdm


from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable
import torch.nn.init as init
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import copy


# Setup CUDA
seed_num = 777
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch.manual_seed(seed_num)
# if device == "cuda:0":
#     torch.cuda.manual_seed_all(seed_num)
# device = "cpu"
device = "cuda:1"
logger, exp_seq = get_logger(filename_prefix="server_")
logger.info(f"-------------------------Session: Exp {exp_seq}")



def send_msg(sock, msg):
    assert isinstance(msg, dict)
    msg['communication_time_stamp'] = time.time()
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
    global total_communication_time
    global offset_time
    total_communication_time += time.time() - msg['communication_time_stamp'] + offset_time
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




resnet_server =  ResNet50_server(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = optim.SGD(resnet_server.parameters(), lr=lr, momentum=0.9)


resnet_server


# host = '10.2.144.188'
# host = '10.9.240.14'
host = '10.2.143.109'
port = 10081

s = socket.socket()
s.bind((host, port))
s.listen(5)

conn, addr = s.accept()
logger.info(f"Connected to: {addr}")

total_communication_time = 0
offset_time = 0
# read epoch
rmsg, data_size = recv_msg(conn) # receive total bach number and epoch from client.
epoch = rmsg['epoch']
num_batch = rmsg['total_batch']
offset_time = - total_communication_time # setting the first communication time as 0 to offset the time.



logger.info(f"received epoch: {rmsg['epoch']}, {rmsg['total_batch']}")

send_msg(conn, {"server_name" : server_name, "server_time": time.time()}) # send server meta information.

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
        msg = {
            "grad": client_output_cpu.grad.clone().detach(),
        }
        data_size = send_msg(conn, msg)
        
        optimizer.step()
        

        if (i + 1) % 100 == 0:

            # measure accuracy and record loss
            _, predicted = torch.max(output, 1)
            correct = (predicted == label).sum().item()
            accuracy = correct / len(label)
            logger.info(f'Epoch: {epc+1}/{epoch}, Batch: {i+1}/{num_batch}, Train Loss: {round(loss.item(), 2)} Train Accuracy: {round(accuracy, 2)}')
            server_to_client_communication_time = recv_msg(conn)[0]['server_to_client_communication_time']
            logger.info(f"Client to server com. time: {round(total_communication_time, 2)}") 
            logger.info(f"Server to client com. time: {round(server_to_client_communication_time, 2)}")

            

    # # validation after each epoch    
    logger.info("Start validation: Sending model to client, who will perform validation.")
    data_size = send_msg(conn, {"server model": resnet_server.state_dict()}) # send model to client.
    
    # rmsg, data_size = recv_msg(conn) # receive total bach number and epoch from client.
    # num_test_batch = rmsg['num_batch']
    # test_dataset_size = rmsg['dataset_size']
    # resnet_server.eval()
    # with torch.no_grad():
    #     logits_all, targets_all = torch.tensor([], device='cpu'), torch.tensor([], dtype=torch.int, device='cpu')
    #     # for j in range(num_test_batch):
    #     for j in range(num_test_batch):
    #         msg, data_size = recv_msg(conn)
    #         # label
    #         label = msg['label']
    #         label = label.clone().detach().long().to(device) # conversion between gpu and cpu.

    #         # feature
    #         client_output_cpu = msg['client_output']
    #         client_output = client_output_cpu.to(device)
            
    #         # forward propagation
    #         logits = resnet_server(client_output)
    #         logits_all = torch.cat((logits_all, logits.detach().cpu()),dim=0)
    #         targets_all = torch.cat((targets_all, label.cpu()), dim=0)

    #     pred = F.log_softmax(logits_all, dim=1)
    #     test_loss = criterion(pred, targets_all)/test_dataset_size # validation loss
        
    #     output = pred.argmax(dim=1) # predicated/output label
    #     prob = F.softmax(logits_all, dim=1) # probabilities

    #     test_acc = accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
    #     test_bal_acc = balanced_accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
    #     test_auc = roc_auc_score(targets_all.numpy(), prob.numpy(), multi_class='ovr')
    #     logger.info(f'Test Loss: {round(test_loss.item(), 2)} Test Accuracy: {round(test_acc, 2)} Test AUC: {round(test_auc, 2)} Test Balanced Accuracy: {round(test_bal_acc, 2)}')

    rmsg = recv_msg(conn)[0]
    logger.info(f'Test Loss: {round(rmsg["Test Loss"], 2)} Test Accuracy: {round(rmsg["Test Accuracy"], 2)} Test AUC: {round(rmsg["Test AUC"], 2)} Test Balanced Accuracy: {round(rmsg["Test Balanced Accuracy"], 2)}')
        

server_to_client_communication_time = recv_msg(conn)[0]['server_to_client_communication_time']        

logger.info(f'Contribution from {server_name} is done')
logger.info(f"Client to server communication time: {round(total_communication_time, 2)}")
logger.info(f"Server to client communication time: {server_to_client_communication_time}")
logger.info(f'Total duration is: {round(time.time() - start_time, 2)} seconds')





