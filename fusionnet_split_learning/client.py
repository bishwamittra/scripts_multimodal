
import struct
import socket
import pickle
import time
from model_client import FusionNet_client
from model_server import FusionNet_server
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import generate_dataloader
from dependency import *
from utils import get_logger


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--connection_start_from_client', action='store_true', default=False)
parser.add_argument('--client_in_sambanova', action='store_true', default=False)
args = parser.parse_args()


logger, exp_seq, save_path = get_logger(filename_prefix="client_")
logger.info(f"-------------------------Session: Exp {exp_seq}")




# Setup cpu
device = 'cpu'
epochs = 10
lr = 3e-5
batch_size = 32
num_workers = 8
device = "cpu"
shape = (224, 224)
torch.manual_seed(777)


# Setup client order
client_order = int(0)
print('Client starts from: ', client_order)


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
    msg = recv_all(sock, msg_len)
    msg = pickle.loads(msg)
    global total_communication_time
    global offset_time
    total_communication_time += time.time() - \
        msg['communication_time_stamp'] + offset_time
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


# data
train_dataloader, val_dataloader, test_dataloader = generate_dataloader(
    shape, batch_size, num_workers)


# Definition of client side model (input layer only)
client_model = FusionNet_client(class_list).to(device)
optimizer = optim.Adam(client_model.parameters(), lr=lr)
lr = 0.001



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


msg = {
    'epoch': epochs,
    'batch_size': batch_size,
    'num_batch': len(train_dataloader)
}

send_msg(s1, msg)  # send 'epoch' and 'batch size' to server


for epc in range(epochs):
    print("running epoch ", epc)
    client_model.set_mode('train')
    target = 0

    for i, (clinic_image, derm_image, meta_data, label) in enumerate(tqdm(train_dataloader, ncols=100, desc='Training with {}'.format(remote_server))):
        optimizer.zero_grad()
        clinic_image = clinic_image.to(device)
        derm_image = derm_image.to(device)
        meta_data = meta_data.to(device)

        output = client_model((clinic_image, derm_image))
        x_clic, x_derm = output
        x_clic = x_clic.clone().detach().requires_grad_(True)
        x_derm = x_derm.clone().detach().requires_grad_(True)

        msg = {
            'label': label,
            'x_clic': x_clic,
            'x_derm': x_derm
        }

        send_msg(s1, msg)  # send label and output(feature) to server
        # receive gradaint after the server has completed the back propagation.
        rmsg = recv_msg(s1)
        x_clic_grad = rmsg['x_clic_grad']
        x_derm_grad = rmsg['x_derm_grad']

        # continue back propagation for client side layers.
        x_clic.backward(x_clic_grad)
        x_derm.backward(x_derm_grad)
        optimizer.step()

    
    # validation
    validation_start_time = time.time()
    rmsg = recv_msg(s1)
    server_model_state_dict = rmsg['server model']
    server_model = FusionNet_server(class_list).to(device)
    server_model.load_state_dict(server_model_state_dict)
    server_model.set_mode('eval')
    client_model.set_mode('eval')
    

send_msg(s1, {'server_to_client_communication_time': total_communication_time})

s1.close()

end_time = time.time()

#
