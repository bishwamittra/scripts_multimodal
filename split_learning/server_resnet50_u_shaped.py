
server_name = 'SERVER_001'

import socket
import struct
import pickle
import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
import time
from model_server_u_shaped import ResNet50 as ResNet50_server
from utils import get_logger
from tqdm import tqdm


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--connection_start_from_client', action='store_true', default=False)
parser.add_argument('--client_in_sambanova', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()


# Setup CUDA

# device = "cpu"
device = "cuda:1"

# seed
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)     


logger, exp_seq = get_logger(filename_prefix="server_")
logger.info(f"-------------------------Session: Exp {exp_seq}")


def sync_time(conn, logger):
    # a back and forth communication to sync time between client and server.

    global epoch_communication_time_client_to_server
    global offset_time
    epoch_communication_time_client_to_server = 0
    offset_time = 0
    send_msg(conn, {"sync_time": "sync request from server"})
    rmsg = recv_msg(conn) 
    logger.info(rmsg['sync_time'])
    
    offset_time = - epoch_communication_time_client_to_server # setting the first communication time as 0 to offset the time.
    epoch_communication_time_client_to_server = 0
    
    

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
    global received_msg_len
    received_msg_len += msg_len
    # read the message data
    msg =  recv_all(sock, msg_len)
    msg = pickle.loads(msg)
    global epoch_communication_time_client_to_server
    global offset_time
    epoch_communication_time_client_to_server += time.time() - msg['communication_time_stamp'] + offset_time
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




epoch_communication_time_client_to_server = 0
offset_time = 0
received_msg_len = 0
if(not args.connection_start_from_client):
    # host = '10.2.144.188'
    # host = '10.9.240.14'
    host = '10.2.143.109'
    port = 10081

    s = socket.socket()
    s.bind((host, port))
    s.listen(5)
    conn, addr = s.accept()
    logger.info(f"Connected to: {addr}")


    # First communication
    rmsg = recv_msg(conn) 
    logger.info(rmsg['initial_msg'])
    send_msg(conn, {"server_name" : server_name}) # send server meta information.


else:
    if(args.client_in_sambanova):
        host = '10.9.240.14'
        port = 8870
    else:
        host = '10.2.143.109'
        port = 10081
    s1 = socket.socket()
    s1.connect((host, port)) # establish connection
    conn = s1
    send_msg(conn, {"initial_msg": "Greetings from Server", "server_name" : server_name})
    rmsg = recv_msg(conn) 
    logger.info(rmsg['initial_msg'])


   
rmsg = recv_msg(conn) # receive total bach number and epoch from client.
epoch = rmsg['epoch']
num_batch = rmsg['total_batch']
lr = rmsg['lr']
logger.info(f"received epoch: {rmsg['epoch']}, batch: {rmsg['total_batch']}")



resnet_server =  ResNet50_server().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_server.parameters(), lr=lr, momentum=0.9)

# Start training
start_time = time.time()
total_validation_time = 0
total_communication_time_client_to_server = 0
total_communication_time_server_to_client = 0
epoch_received_msg_len = 0
total_size_client_head_output = 0
total_size_server_gradient = 0

training_time = 0
logger.info(f"Start training @ {time.asctime()}")

for epc in range(epoch):
    sync_time(conn, logger)
    epoch_start_time = time.time()
    epoch_training_time = 0
    epoch_size_client_head_output = 0
    epoch_size_server_gradient = 0
    # epoch_communication_time_client_to_server = 0
    # offset_time, epoch_communication_time_client_to_server = sync_time(offset_time, epoch_communication_time_client_to_server)
    

    
    resnet_server.train()
    for i in tqdm(range(num_batch), ncols = 100, desc='Training with {}'.format(server_name)):
        batch_training_start_time = time.time()
        optimizer.zero_grad()
        epoch_training_time += time.time() - batch_training_start_time
        
        size_client_head_output = received_msg_len
        rmsg = recv_msg(conn) # receives feature from client.
        size_client_head_output = received_msg_len - size_client_head_output
        epoch_size_client_head_output += size_client_head_output
        

        # forward propagation
        batch_training_start_time = time.time()
        client_output_cpu = rmsg['client_output'] # feature
        client_output = client_output_cpu.to(device)
        server_output_gpu = resnet_server(client_output)
        server_output = server_output_gpu.cpu().clone().detach().requires_grad_(True)
        msg = {
            "server_output": server_output,
        }
        epoch_training_time += time.time() - batch_training_start_time
        
        
        send_msg(conn, msg) # send server output to client
        size_server_gradient = received_msg_len
        rmsg = recv_msg(conn) # receive gradient from client
        size_server_gradient = received_msg_len - size_server_gradient
        epoch_size_server_gradient += size_server_gradient

        # backward propagation
        batch_training_start_time = time.time()
        server_grad = rmsg['server_grad'].to(device)
        server_output_gpu.backward(server_grad)
        epoch_training_time += time.time() - batch_training_start_time
        
        
        
        # send gradient to client
        batch_training_start_time = time.time()
        msg = {
            "client_grad": client_output_cpu.grad.clone().detach(),
        }
        epoch_training_time += time.time() - batch_training_start_time
        send_msg(conn, msg)


        optimizer.step()
        
        
        

        if (i + 1) % 100 == 0:
            pass
            # break

         

    # validation after each epoch    
    data_size = send_msg(conn, {"server model": {k: v.cpu() for k, v in resnet_server.state_dict().items()}}) # send model to client.
    rmsg = recv_msg(conn)
    logger.info("")
    logger.info(f"Epoch {epc+1}/{epoch} results:")
    # logger.info(f"Train Loss: {round(rmsg['Train Loss'], 4)} Train Accuracy: {round(rmsg['Train Accuracy'], 4)} Train AUC: {round(rmsg['Train AUC'], 4)} Train Balanced Accuracy: {round(rmsg['Train Balanced Accuracy'], 4)}")
    logger.info(f'Test Loss: {round(rmsg["Test Loss"], 4)} Test Accuracy: {round(rmsg["Test Accuracy"], 4)} Test AUC: {round(rmsg["Test AUC"], 4)} Test Balanced Accuracy: {round(rmsg["Test Balanced Accuracy"], 4)}')
    
    
    # show time
    epoch_communication_time_server_to_client = recv_msg(conn)['server_to_client_communication_time']
    logger.info(f"Epoch: client to server com. time: {round(epoch_communication_time_client_to_server, 2)}") 
    logger.info(f"Epoch: server to client com. time: {round(epoch_communication_time_server_to_client, 2)}")
    total_communication_time_server_to_client += epoch_communication_time_server_to_client
    send_msg(conn, {'client_to_server_communication_time': epoch_communication_time_client_to_server})
    logger.info(f"Epoch: training time server: {round(epoch_training_time, 2)}")
    training_time += epoch_training_time  
    total_validation_time += rmsg['validation time']
    logger.info(f"Epoch: validation time: {round(rmsg['validation time'], 2)}")
    logger.info(f"Epoch: total time: {round(time.time() - epoch_start_time, 2)}")
    total_communication_time_client_to_server += epoch_communication_time_client_to_server


    logger.info("")
    logger.info(f"Epoch: received msg len from client: {round((received_msg_len - epoch_received_msg_len)/1024/1024, 2)} MB")
    logger.info(f"Epoch: size of client head output: {round(epoch_size_client_head_output/1024/1024, 2)} MB")
    logger.info(f"Epoch: size of server gradient: {round(epoch_size_server_gradient/1024/1024, 2)} MB")
    total_size_client_head_output += epoch_size_client_head_output
    total_size_server_gradient += epoch_size_server_gradient
    epoch_received_msg_len = received_msg_len
    

total_communication_time_server_to_client = recv_msg(conn)['server_to_client_communication_time']        
send_msg(conn, {'client_to_server_communication_time': total_communication_time_client_to_server})


logger.info("")
logger.info(f'Summary')
logger.info(f"Client to server communication time: {round(total_communication_time_client_to_server, 2)}")
logger.info(f"Server to client communication time: {round(total_communication_time_server_to_client, 2)}")
logger.info(f"Training time server: {round(training_time, 2)}")
logger.info(f"Validation time: {round(total_validation_time, 2)}")
logger.info(f'Total duration is: {round(time.time() - start_time, 2)} seconds')
logger.info("")
logger.info(f"Received msg len from client: {round(received_msg_len/1024/1024, 2)} MB")
logger.info(f"Total size of client head output: {round(total_size_client_head_output/1024/1024, 2)} MB")
logger.info(f"Total size of server gradient: {round(total_size_server_gradient/1024/1024, 2)} MB")




