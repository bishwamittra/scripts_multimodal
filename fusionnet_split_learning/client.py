
import struct
import socket
import pickle
import time
from model_client import FusionNet
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import generate_dataloader
from dependency import *


# Setup cpu
device = 'cpu'
epochs = 10
lr = 3e-5
batch_size = 32
num_workers = 8
device = "cuda:0"
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



# data
train_dataloader, val_dataloader, test_dataloader = generate_dataloader(shape, batch_size, num_workers)


# Definition of client side model (input layer only)
client_model = FusionNet(class_list).to(device)
optimizer = optim.Adam(client_model.parameters(), lr=lr)

lr = 0.001


# Training 



# connection to server
# host = '10.2.144.188'
# host = '10.9.240.14'
host = '10.2.143.109'
port = 10081
s1 = socket.socket()
s1.connect((host, port)) # establish connection
# s1.close()


start_time = time.time()


msg = {
    'epoch': epochs,
    'total_batch': len(train_dataloader)
}

send_msg(s1, msg) # send 'epoch' and 'batch size' to server

# resnet_client.eval() # Why eval()?
total_communication_time = 0
offset_time = 0
remote_server = recv_msg(s1)['server_name'] # get server's meta information.
offset_time = - total_communication_time


for epc in range(epochs):
    print("running epoch ", epc)

    target = 0

    for i, (clinic_image, derm_image, meta_data, label) in enumerate(tqdm(train_dataloader, ncols=100, desc='Training with {}'.format(remote_server))):
        optimizer.zero_grad()
        clinic_image = clinic_image.to(device)
        derm_image = derm_image.to(device)
        meta_data = meta_data.to(device)

        output = client_model((clinic_image, derm_image))
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


        # if (i + 1) % 1000 == 0:
        #     msg = {
        #         'num_batch': len(test_loader),
        #         'dataset_size': len(test_loader.dataset)
        #     }
        #     # print(msg)
        #     send_msg(s1, msg) # 'num test batch' to server
        #     resnet_client.eval()
        #     with torch.no_grad():
        #         for x, label in test_loader:
        #             x = x.to(device)
        #             label = label.to(device)
        #             output = resnet_client(x)
        #             client_output = output.clone().detach().requires_grad_(True)
        #             msg = {
        #                 'label': label,
        #                 'client_output': client_output
        #             }
        #             send_msg(s1, msg) # send label and output(feature) to server

        #     # break
        #     send_msg(s1, {'server_to_client_communication_time': round(total_communication_time, 2)})       
        #     # break
        
        # if(i+1) % 100 == 0:
        #     print(f"Server to client communication time: {round(total_communication_time, 2)}")
            


send_msg(s1, {'server_to_client_communication_time': total_communication_time})       
     
s1.close()

end_time = time.time()

# 





