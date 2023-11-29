
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
from utils import get_logger, compress, decompress
import time
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from model_server_u_shaped import FusionNet_server_middle
from dependency import class_list
import torch
import numpy as np
import random
import pickle
import struct
import socket
server_name = 'SERVER_001'





import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--connection_start_from_client', action='store_true', default=False)
parser.add_argument('--client_in_sambanova', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--save_root', type=str, default='result_dump', help='root path to save results')
args = parser.parse_args()



# Setup CUDA
device = "cuda:1"
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)     
logger, exp_seq, save_path = get_logger(save_root=args.save_root, filename_prefix="server_u_shaped_compressed_")
logger.info(f"-------------------------Session: Exp {exp_seq}")



def sync_time(conn, logger):
    # a back and forth communication to sync time between client and server.

    global epoch_communication_time_client_to_server
    global offset_time
    epoch_communication_time_client_to_server = 0
    offset_time = 0
    send_msg(conn, {"sync_time": "sync request from server"})
    rmsg = recv_msg(conn) 
    # logger.info(rmsg['sync_time'])
    
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
    msg = recv_all(sock, msg_len)
    msg = pickle.loads(msg)
    global epoch_communication_time_client_to_server
    global offset_time
    epoch_communication_time_client_to_server += time.time() - \
        msg['communication_time_stamp'] + offset_time
    # print("\n", epoch_communication_time_client_to_server)
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
    print(rmsg['initial_msg'])
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
    print(rmsg['initial_msg'])
    

rmsg = recv_msg(conn)
epochs = rmsg['epoch']
num_batch = rmsg['num_batch']
lr = rmsg['lr']
cd_method = rmsg['cd_method']
architecture_choice = rmsg['architecture_choice']
logger.info(f"received epoch: {rmsg['epoch']}, num_batch: {rmsg['num_batch']}, learning rate: {rmsg['lr']}, architecture_choice: {rmsg['architecture_choice']}, cd_method: {rmsg['cd_method']}")


server_model = FusionNet_server_middle(architecture_choice=architecture_choice).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(server_model.parameters(), lr=lr)


# Start training
start_time = time.time()
total_validation_time = 0
total_test_time = 0
total_training_time = 0
total_communication_time_client_to_server = 0
total_communication_time_server_to_client = 0
epoch_received_msg_len = 0
total_size_client_head_output = 0
total_size_server_gradient = 0




logger.info(f"Start training @ {time.asctime()}")
server_model.set_mode('train')
for epc in range(epochs):
    sync_time(conn, logger)
    epoch_start_time = time.time()
    epoch_compresson_decompression_time = 0
    epoch_training_time = 0
    epoch_size_client_head_output = 0
    epoch_size_server_gradient = 0
    

    for index in tqdm(range(num_batch), ncols=100, desc='Training with {}'.format(server_name)):
        batch_training_start_time = time.time()
        optimizer.zero_grad()
        epoch_training_time += time.time() - batch_training_start_time

        size_client_head_output = received_msg_len
        rmsg = recv_msg(conn) # receives label and feature from client.
        size_client_head_output = received_msg_len - size_client_head_output
        epoch_size_client_head_output += size_client_head_output


        epoch_compresson_decompression_start_time = time.time()
        x_clic_cpu, x_derm_cpu = rmsg['x_clic'], rmsg['x_derm']
        x_clic_cpu = torch.from_numpy(np.frombuffer(decompress(x_clic_cpu, cd_method=cd_method), dtype=np.float32).reshape(rmsg['x_clic_shape'])).requires_grad_(True) 
        x_derm_cpu = torch.from_numpy(np.frombuffer(decompress(x_derm_cpu, cd_method=cd_method), dtype=np.float32).reshape(rmsg['x_derm_shape'])).requires_grad_(True)
        x_clic = x_clic_cpu.to(device)
        x_derm = x_derm_cpu.to(device)
        epoch_compresson_decompression_time += time.time() - epoch_compresson_decompression_start_time

        # forward propagation
        batch_training_start_time = time.time()
        x_clic_server_gpu, x_derm_server_gpu, x_fusion_server_gpu = server_model((x_clic, x_derm))
        epoch_training_time += time.time() - batch_training_start_time

        epoch_compresson_decompression_start_time = time.time()
        x_clic_server = x_clic_server_gpu.cpu().clone().detach()
        x_derm_server = x_derm_server_gpu.cpu().clone().detach()
        x_fusion_server = x_fusion_server_gpu.cpu().clone().detach()
        msg = {
            'x_clic_server' : compress(x_clic_server.numpy(), cd_method=cd_method),
            'x_clic_server_shape' : x_clic_server.shape,
            'x_derm_server' : compress(x_derm_server.numpy(), cd_method=cd_method),
            'x_derm_server_shape' : x_derm_server.shape,
            'x_fusion_server' : compress(x_fusion_server.numpy(), cd_method=cd_method),
            'x_fusion_server_shape' : x_fusion_server.shape,
        }
        epoch_compresson_decompression_time += time.time() - epoch_compresson_decompression_start_time
        
        
        send_msg(conn, msg) # send server output to client
        size_server_gradient = received_msg_len
        rmsg = recv_msg(conn) # receive gradient from client
        size_server_gradient = received_msg_len - size_server_gradient
        epoch_size_server_gradient += size_server_gradient

        epoch_compresson_decompression_start_time = time.time()
        x_clic_server_grad = torch.from_numpy(np.frombuffer(decompress(rmsg['x_clic_server_grad'], cd_method=cd_method), dtype=np.float32).reshape(rmsg['x_clic_server_grad_shape'])).to(device)
        x_derm_server_grad = torch.from_numpy(np.frombuffer(decompress(rmsg['x_derm_server_grad'], cd_method=cd_method), dtype=np.float32).reshape(rmsg['x_derm_server_grad_shape'])).to(device)
        # x_fusion_server_grad = torch.from_numpy(np.frombuffer(decompress(rmsg['x_fusion_server_grad']), dtype=np.float32).reshape(rmsg['x_fusion_server_grad_shape'])).to(device)
        epoch_compresson_decompression_time += time.time() - epoch_compresson_decompression_start_time
    

        # backward propagation
        batch_training_start_time = time.time()
        x_clic_server_gpu.backward(x_clic_server_grad.to(device))
        x_derm_server_gpu.backward(x_derm_server_grad.to(device))
        epoch_training_time += time.time() - batch_training_start_time
        
        # Interesting error: Trying to backward through the graph a second time
        # x_fusion_server_gpu.backward(rmsg['x_fusion_server_grad'].to(device))
        # send gradient to client

        epoch_compresson_decompression_start_time = time.time()
        msg = {
            "x_clic_grad": compress(x_clic_cpu.grad.clone().detach().numpy(), cd_method=cd_method),
            "x_clic_grad_shape": x_clic_cpu.grad.shape,
            "x_derm_grad": compress(x_derm_cpu.grad.clone().detach().numpy(), cd_method=cd_method),
            "x_derm_grad_shape": x_derm_cpu.grad.shape,
        }
        epoch_compresson_decompression_time += time.time() - epoch_compresson_decompression_start_time
        
        send_msg(conn, msg)

        batch_training_start_time = time.time()
        optimizer.step()
        epoch_training_time += time.time() - batch_training_start_time
        

        # break

    total_training_time += epoch_training_time

    # for validation and test, send server model to client
    epoch_compresson_decompression_start_time = time.time()
    _server_model = {}
    for k, v in server_model.state_dict().items():
        if('num_batches_tracked' in k):
            continue
        v = v.cpu().clone().detach().numpy()
        _server_model[k] = (compress(v, cd_method=cd_method), v.shape)
    msg = {
        "server_model_state_dict": _server_model,
        "server_training_time": epoch_training_time
    }
    epoch_compresson_decompression_time += time.time() - epoch_compresson_decompression_start_time

    send_msg(conn, msg) # send model to client.
    

    # logging
    logger.info("")
    logger.info(f"Epoch {epc+1}/{epochs} results:")
    
    # validation
    rmsg = recv_msg(conn)
    logger.info(f"Validation loss: {round(rmsg['validation loss'], 4)}, Validation dia acc: {round(rmsg['validation dia acc'], 4)}, Validation sps acc: {round(rmsg['validation sps acc'], 4)}, Validation mean acc: {round(rmsg['validation mean acc'], 4)}")
    validation_time = rmsg['validation time']
    total_validation_time += validation_time
    

    # test
    rmsg = recv_msg(conn)
    if(rmsg['is_best_val']):
        logger.info(f"Test loss: {round(rmsg['test loss'], 4)}, Test dia acc: {round(rmsg['test dia acc'], 4)}, Test sps acc: {round(rmsg['test sps acc'], 4)}, Test mean acc: {round(rmsg['test mean acc'], 4)}")
        test_time = rmsg['test time']
    else:
        test_time = 0
    total_test_time += test_time

    # show time
    logger.info("")
    epoch_communication_time_server_to_client = recv_msg(conn)['server_to_client_communication_time']
    logger.info(f"Epoch: client to server com. time: {round(epoch_communication_time_client_to_server, 2)}") 
    logger.info(f"Epoch: server to client com. time: {round(epoch_communication_time_server_to_client, 2)}")
    total_communication_time_server_to_client += epoch_communication_time_server_to_client
    send_msg(conn, {'client_to_server_communication_time': epoch_communication_time_client_to_server})
    
    
    logger.info(f"Epoch: training time server: {round(epoch_training_time, 2)}")
    logger.info(f"Epoch: validation time: {round(validation_time, 2)}")
    logger.info(f"Epoch: test time: {round(test_time, 2)}")
    # logger.info(f"Epoch: server encode time: {round(server_encode_time, 2)}")
    logger.info(f"Epoch: server compresson_decompression time: {round(epoch_compresson_decompression_time, 2)}")
    logger.info(f"Epoch: total time: {round(time.time() - epoch_start_time, 2)}")
    total_communication_time_client_to_server += epoch_communication_time_client_to_server


    logger.info("")
    logger.info(f"Epoch: received msg len from client: {round((received_msg_len - epoch_received_msg_len)/1024/1024, 2)} MB")
    logger.info(f"Epoch: size of client head output: {round(epoch_size_client_head_output/1024/1024, 2)} MB")
    logger.info(f"Epoch: size of server gradient: {round(epoch_size_server_gradient/1024/1024, 2)} MB")

    msg = {
        'size_client_to_server_msg': received_msg_len - epoch_received_msg_len,
        'size_client_head_output': epoch_size_client_head_output,
        'size_server_gradient': epoch_size_server_gradient
    }
    send_msg(conn, msg)

    total_size_client_head_output += epoch_size_client_head_output
    total_size_server_gradient += epoch_size_server_gradient
    epoch_received_msg_len = received_msg_len
    


total_communication_time_server_to_client = recv_msg(conn)['server_to_client_communication_time']        
send_msg(conn, {'client_to_server_communication_time': total_communication_time_client_to_server})


logger.info("")
logger.info(f'Summary')
logger.info(f"Client to server communication time: {round(total_communication_time_client_to_server, 2)}")
logger.info(f"Server to client communication time: {round(total_communication_time_server_to_client, 2)}")
logger.info(f"Training time server: {round(total_training_time, 2)}")
logger.info(f"Validation time: {round(total_validation_time, 2)}")
logger.info(f"Test time: {round(total_test_time, 2)}")
logger.info(f'Total duration is: {round(time.time() - start_time, 2)} seconds')
logger.info("")
logger.info(f"Received msg len from client: {round(received_msg_len/1024/1024, 2)} MB")
logger.info(f"Total size of client head output: {round(total_size_client_head_output/1024/1024, 2)} MB")
logger.info(f"Total size of server gradient: {round(total_size_server_gradient/1024/1024, 2)} MB")

    


    