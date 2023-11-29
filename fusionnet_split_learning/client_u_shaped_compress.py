
import struct
import socket
import pickle
import time
import pandas as pd
from model_client_u_shaped import FusionNet_client_first, FusionNet_client_last
from model_server_u_shaped import FusionNet_server_middle
from tqdm import tqdm
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import generate_dataloader
from dependency import *
from utils import get_logger, compress, decompress
from utils_client import validation_u_shaped
from sys import getsizeof


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--connection_start_from_client', action='store_true', default=False)
parser.add_argument('--epoch', type=int, default=2, help='Number of epochs')
parser.add_argument('--architecture_choice', type=int, default=1, help='Index of architecture choice')
parser.add_argument('--client_in_sambanova', action='store_true', default=False)
parser.add_argument('--save_root', type=str, default='result_dump', help='root path to save results')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()


logger, exp_seq, save_path = get_logger(save_root=args.save_root, filename_prefix="client_u_shaped_compressed_")
logger.info(f"-------------------------Session: Exp {exp_seq}")




# Setup cpu
device = 'cpu'
epochs = args.epoch
cd_method = 'zstd'
# cd_method = 'blosc'
lr = 0.001
# lr = 3e-5
batch_size = 32
num_workers = 8
shape = (224, 224)
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)     



# Setup client order
client_order = int(0)
print('Client starts from: ', client_order)


def sync_time(conn, logger):
    # a back and forth communication to sync time between client and server.

    global epoch_communication_time_server_to_client
    global offset_time
    epoch_communication_time_server_to_client = 0
    offset_time = 0
    rmsg = recv_msg(conn)
    # logger.info(rmsg['sync_time'])
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
    msg = recv_all(sock, msg_len)
    msg = pickle.loads(msg)
    global epoch_communication_time_server_to_client
    global offset_time
    epoch_communication_time_server_to_client += time.time() - \
        msg['communication_time_stamp'] + offset_time
    # print("\n", epoch_communication_time_server_to_client)
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
client_model_first = FusionNet_client_first(architecture_choice=args.architecture_choice).to(device)
optimizer_first = optim.Adam(client_model_first.parameters(), lr=lr)

client_model_last = FusionNet_client_last(class_list, architecture_choice=args.architecture_choice).to(device)
optimizer_last = optim.Adam(client_model_last.parameters(), lr=lr)



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
total_training_time = 0
total_server_training_time = 0
total_validation_time = 0
total_test_time = 0
total_communication_time_server_to_client = 0
total_communication_time_client_to_server = 0
epoch_received_msg_len = 0
total_size_server_model = 0
total_size_server_output = 0
total_size_client_first_gradient = 0
best_mean_acc = 0

msg = {
    'epoch': epochs,
    'num_batch': len(train_dataloader),
    'lr': lr,
    'architecture_choice': args.architecture_choice,
    'cd_method': cd_method,
}


logger.info(f"Epoch: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}, Architecture Choice: {args.architecture_choice}")

send_msg(s1, msg)  # send 'epoch' and 'batch size' to server

best_val_mean_acc = 0


for epc in range(epochs):
    entry = {}
    print("running epoch ", epc)
    sync_time(s1, logger)
    epoch_start_time = time.time()
    epoch_training_time = 0
    epoch_training_time_server = 0
    epoch_size_server_output = 0
    epoch_size_client_first_gradient = 0
    epoch_communication_time_server_to_client = 0
    epoch_compresson_decompression_time = 0
    is_best_val = False


    train_loss = 0
    train_dia_acc = 0
    train_sps_acc = 0

    client_model_first.set_mode('train')
    client_model_last.set_mode('train')
    for i, (clinic_image, derm_image, meta_data, label) in enumerate(tqdm(train_dataloader, ncols=100, desc='Training with {}'.format(remote_server))):
        
        # client_first model forward propagation
        batch_training_start_time = time.time()
        optimizer_first.zero_grad()
        optimizer_last.zero_grad()
        clinic_image = clinic_image.to(device)
        derm_image = derm_image.to(device)
        meta_data = meta_data.to(device)


       

        output_first = client_model_first((clinic_image, derm_image))
        epoch_training_time += time.time() - batch_training_start_time

        epoch_compresson_decompression_start_time = time.time()
        x_clic, x_derm = output_first
        # x_clic = x_clic.clone().detach().requires_grad_(True)
        # x_derm = x_derm.clone().detach().requires_grad_(True)
        x_clic = x_clic.clone().detach()
        x_derm = x_derm.clone().detach()
        # msg = {
        #     'x_clic': x_clic,
        #     'x_derm': x_derm
        # }
        msg = {
            'x_clic': compress(x_clic.numpy(), cd_method=cd_method),
            'x_clic_shape': x_clic.shape,
            'x_derm': compress(x_derm.numpy(), cd_method=cd_method),
            'x_derm_shape': x_derm.shape
        }
        x_clic = x_clic.requires_grad_(True)
        x_derm = x_derm.requires_grad_(True)
        epoch_compresson_decompression_time += time.time() - epoch_compresson_decompression_start_time

        

        send_msg(s1, msg)  # send label and output(feature) to server

        size_server_output = received_msg_len
        rmsg = recv_msg(s1) # receive gradaint after the server has completed the back propagation.
        size_server_output = received_msg_len - size_server_output
        epoch_size_server_output += size_server_output


       
        epoch_compresson_decompression_start_time = time.time()
        x_clic_server_gpu = rmsg['x_clic_server']
        x_derm_server_gpu = rmsg['x_derm_server']
        x_fusion_server_gpu = rmsg['x_fusion_server']
        # convert
        x_clic_server_gpu = torch.from_numpy(np.frombuffer(decompress(x_clic_server_gpu, cd_method=cd_method), dtype=np.float32).reshape(rmsg['x_clic_server_shape'])).requires_grad_(True)
        x_derm_server_gpu = torch.from_numpy(np.frombuffer(decompress(x_derm_server_gpu, cd_method=cd_method), dtype=np.float32).reshape(rmsg['x_derm_server_shape'])).requires_grad_(True)
        x_fusion_server_gpu = torch.from_numpy(np.frombuffer(decompress(x_fusion_server_gpu, cd_method=cd_method), dtype=np.float32).reshape(rmsg['x_fusion_server_shape'])).requires_grad_(True) 
        # to device
        x_clic_server = x_clic_server_gpu.to(device)
        x_derm_server = x_derm_server_gpu.to(device)
        x_fusion_server = x_fusion_server_gpu.to(device)
        epoch_compresson_decompression_time += time.time() - epoch_compresson_decompression_start_time
        
        # client_last model forward propagation and loss calculation
        batch_training_start_time = time.time()
        loss, dia_acc, _,  sps_acc, _, _, _ = client_model_last.forward_propagate_and_loss_compute((x_clic_server, x_derm_server, x_fusion_server), label, batch_size, device)        
        train_loss += loss.item()
        train_dia_acc += dia_acc.item()
        train_sps_acc += sps_acc.item()
        loss.backward()
        epoch_training_time += time.time() - batch_training_start_time


        # send gradient to server
        epoch_compresson_decompression_start_time = time.time()
        msg = {
            "x_clic_server_grad": compress(x_clic_server_gpu.grad.clone().detach().numpy(), cd_method=cd_method),
            "x_clic_server_grad_shape": x_clic_server_gpu.grad.shape,
            "x_derm_server_grad": compress(x_derm_server_gpu.grad.clone().detach().numpy(), cd_method=cd_method),
            "x_derm_server_grad_shape": x_derm_server_gpu.grad.shape,
            # "x_fusion_server_grad": compress(x_fusion_server_gpu.grad.clone().detach().numpy()),
            # "x_fusion_server_grad_shape": x_fusion_server_gpu.grad.shape
        }
        epoch_compresson_decompression_time += time.time() - epoch_compresson_decompression_start_time

        send_msg(s1, msg)

        batch_training_start_time = time.time()
        optimizer_last.step()
        epoch_training_time += time.time() - batch_training_start_time


        # back propagation client_first model
        size_client_first_gradient = received_msg_len
        rmsg = recv_msg(s1) # receive gradient of the head from server
        size_client_first_gradient = received_msg_len - size_client_first_gradient
        epoch_size_client_first_gradient += size_client_first_gradient


        epoch_compresson_decompression_start_time = time.time()
        x_clic_grad = torch.from_numpy(np.frombuffer(decompress(rmsg['x_clic_grad'], cd_method=cd_method), dtype=np.float32).reshape(rmsg['x_clic_grad_shape'])).requires_grad_(True)
        x_derm_grad = torch.from_numpy(np.frombuffer(decompress(rmsg['x_derm_grad'], cd_method=cd_method), dtype=np.float32).reshape(rmsg['x_derm_grad_shape'])).requires_grad_(True)
        epoch_compresson_decompression_time += time.time() - epoch_compresson_decompression_start_time
        
        batch_training_start_time = time.time()
        x_clic.backward(x_clic_grad)
        x_derm.backward(x_derm_grad)
        optimizer_first.step()
        epoch_training_time += time.time() - batch_training_start_time

        # break

    train_loss = train_loss / (i + 1)
    train_dia_acc = train_dia_acc / (i + 1)
    train_sps_acc = train_sps_acc / (i + 1)
    total_training_time += epoch_training_time

    train_mean_acc = (train_dia_acc*1 + train_sps_acc*7)/8

    # logging
    logger.info("")
    logger.info(f"Epoch {epc+1}/{epochs} results:")
    logger.info(f"Train Loss: {round(train_loss, 4)}, Train Dia Acc: {round(train_dia_acc, 4)}, Train SPS Acc: {round(train_sps_acc, 4)}, Train Mean Acc: {round(train_mean_acc, 4)}")
    

    # validation
    server_model_size = received_msg_len
    rmsg = recv_msg(s1)
    server_model_size = received_msg_len - server_model_size
    total_size_server_model += server_model_size
    
    epoch_compresson_decompression_start_time = time.time()
    server_model_state_dict = {}
    for k, v in rmsg['server_model_state_dict'].items():
        try:
            server_model_state_dict[k] = torch.from_numpy(np.frombuffer(decompress(v[0], cd_method=cd_method), dtype=np.float32).reshape(v[1]))
        except Exception as e:
            
            print(e)
            print(k)
            print(decompress(v[0], cd_method=cd_method))
            print(v[1])
            quit()
    # server_model_state_dict = {k: torch.from_numpy(np.frombuffer(decompress(v[0], cd_method=cd_method), dtype=np.float32).reshape(v[1])) for k, v in server_model_state_dict.items()}
    epoch_compresson_decompression_time += time.time() - epoch_compresson_decompression_start_time
    
    epoch_server_training_time = rmsg['server_training_time']
    total_server_training_time += epoch_server_training_time
    
    server_load_start_time = time.time()
    server_model = FusionNet_server_middle(architecture_choice=args.architecture_choice).to(device)
    server_model.load_state_dict(server_model_state_dict)
    server_load_time = time.time() - server_load_start_time

    # validation mode
    validation_start_time = time.time()
    # val_loss, val_dia_acc, val_sps_acc = validation_u_shaped(client_model_first, client_model_last, server_model, val_dataloader, device)
    val_loss, \
    val_dia_acc, \
    [val_dia_clic_acc, val_dia_derm_acc, val_dia_fusion_acc], \
    val_sps_acc, \
    [val_sps_clic_acc, val_sps_derm_acc, val_sps_fusion_acc], \
    [val_pn_acc, val_str_acc, val_pig_acc, val_rs_acc, val_dag_acc, val_bwv_acc, val_vs_acc], \
    [
        [val_pn_clic_acc, val_pn_derm_acc, val_pn_fusion_acc], 
        [val_str_clic_acc, val_str_derm_acc, val_str_fusion_acc], 
        [val_pig_clic_acc, val_pig_derm_acc, val_pig_fusion_acc], 
        [val_rs_clic_acc, val_rs_derm_acc, val_rs_fusion_acc], 
        [val_dag_clic_acc, val_dag_derm_acc, val_dag_fusion_acc], 
        [val_bwv_clic_acc, val_bwv_derm_acc, val_bwv_fusion_acc], 
        [val_vs_clic_acc, val_vs_derm_acc, val_vs_fusion_acc]
    ] = validation_u_shaped(client_model_first, client_model_last, server_model, val_dataloader, device)
    
    val_mean_acc = (val_dia_acc * 1 + val_sps_acc * 7) / 8
    validation_time = time.time() - validation_start_time
    msg = {
            'validation loss': val_loss,
            'validation dia acc': val_dia_acc,
            'validation sps acc': val_sps_acc,
            'validation mean acc': val_mean_acc,
            'validation time': validation_time,
    }
    total_validation_time += msg['validation time']

    send_msg(s1, msg)

    

    



    # logger.info(f'Valid Loss: {round(val_loss, 4)}, Valid Dia Acc: {round(val_dia_acc, 4)}, Valid SPS Acc: {round(val_sps_acc, 4)} Valid Mean Acc: {round(val_mean_acc, 4)}')
    logger.info(f'Valid Loss: {round(val_loss, 4)}, Valid Dia Acc: {round(val_dia_acc, 4)}, Valid SPS Acc: {round(val_sps_acc, 4)} Valid Mean Acc: {round(val_mean_acc, 4)}')
    # breakdown accuracy
    logger.info(f'Valid Dia Acc breakdown: mean {round(val_dia_acc, 4)} clic {round(val_dia_clic_acc, 4)}, derm {round(val_dia_derm_acc, 4)}, fusion {round(val_dia_fusion_acc, 4)}')
    logger.info(f'Valid PN Acc breakdown: mean {round(val_pn_acc, 4)}, clic {round(val_pn_clic_acc, 4)}, derm {round(val_pn_derm_acc, 4)}, fusion {round(val_pn_fusion_acc, 4)}')
    logger.info(f'Valid STR Acc breakdown: mean {round(val_str_acc, 4)}, clic {round(val_str_clic_acc, 4)}, derm {round(val_str_derm_acc, 4)}, fusion {round(val_str_fusion_acc, 4)}')
    logger.info(f'Valid PIG Acc breakdown: mean {round(val_pig_acc, 4)}, clic {round(val_pig_clic_acc, 4)}, derm {round(val_pig_derm_acc, 4)}, fusion {round(val_pig_fusion_acc, 4)}')
    logger.info(f'Valid RS Acc breakdown: mean {round(val_rs_acc, 4)}, clic {round(val_rs_clic_acc, 4)}, derm {round(val_rs_derm_acc, 4)}, fusion {round(val_rs_fusion_acc, 4)}')
    logger.info(f'Valid DAG Acc breakdown: mean {round(val_dag_acc, 4)}, clic {round(val_dag_clic_acc, 4)}, derm {round(val_dag_derm_acc, 4)}, fusion {round(val_dag_fusion_acc, 4)}')
    logger.info(f'Valid BWV Acc breakdown: mean {round(val_bwv_acc, 4)}, clic {round(val_bwv_clic_acc, 4)}, derm {round(val_bwv_derm_acc, 4)}, fusion {round(val_bwv_fusion_acc, 4)}')
    logger.info(f'Valid VS Acc breakdown: mean {round(val_vs_acc, 4)}, clic {round(val_vs_clic_acc, 4)}, derm {round(val_vs_derm_acc, 4)}, fusion {round(val_vs_fusion_acc, 4)}')
    logger.info(f'Thus, valid SPS Acc breakdown: mean {round(val_sps_acc, 4)}, clic {round(val_sps_clic_acc, 4)}, derm {round(val_sps_derm_acc, 4)}, fusion {round(val_sps_fusion_acc, 4)}')    
    

    # save the best model
    if val_mean_acc > best_mean_acc:
        best_mean_acc = val_mean_acc
        is_best_val = True
        # torch.save(client_model_first.state_dict(), f'{save_path}/checkpoint/fusionnet_first_stage_client_first.pth')
        # torch.save(server_model.state_dict(), f'{save_path}/checkpoint/fusionnet_first_stage_server_middle.pth')
        # torch.save(client_model_last.state_dict(), f'{save_path}/checkpoint/fusionnet_first_stage_client_last.pth') 
        logger.info(f'Current Best Mean Validation Acc is {round(best_mean_acc, 4)}')
    logger.info("")

    if(is_best_val):
        # test mode
        test_start_time = time.time()
        test_loss, \
        test_dia_acc, \
        [test_dia_clic_acc, test_dia_derm_acc, test_dia_fusion_acc], \
        test_sps_acc, \
        [test_sps_clic_acc, test_sps_derm_acc, test_sps_fusion_acc], \
        [test_pn_acc, test_str_acc, test_pig_acc, test_rs_acc, test_dag_acc, test_bwv_acc, test_vs_acc], \
        [
            [test_pn_clic_acc, test_pn_derm_acc, test_pn_fusion_acc], 
            [test_str_clic_acc, test_str_derm_acc, test_str_fusion_acc], 
            [test_pig_clic_acc, test_pig_derm_acc, test_pig_fusion_acc], 
            [test_rs_clic_acc, test_rs_derm_acc, test_rs_fusion_acc], 
            [test_dag_clic_acc, test_dag_derm_acc, test_dag_fusion_acc], 
            [test_bwv_clic_acc, test_bwv_derm_acc, test_bwv_fusion_acc], 
            [test_vs_clic_acc, test_vs_derm_acc, test_vs_fusion_acc]
        ] = validation_u_shaped(client_model_first, client_model_last, server_model, test_dataloader, device)

        # test_loss, test_dia_acc, test_sps_acc = validation_u_shaped(client_model_first, client_model_last, server_model, test_dataloader, device)
        test_mean_acc = (test_dia_acc*1 + test_sps_acc*7)/8
        test_time = time.time() - test_start_time
        msg = {
                'is_best_val': is_best_val,
                'test loss': test_loss,
                'test dia acc': test_dia_acc,
                'test sps acc': test_sps_acc,
                'test mean acc': test_mean_acc,
                'test time': test_time,
        }
        logger.info(f'Test Loss: {round(test_loss, 4)}, Test Dia Acc: {round(test_dia_acc, 4)}, Test SPS Acc: {round(test_sps_acc, 4)} Test Mean Acc: {round(test_mean_acc, 4)}')
        # breakdown accuracy
        logger.info(f'Test Dia Acc breakdown: mean {round(test_dia_acc, 4)} clic {round(test_dia_clic_acc, 4)}, derm {round(test_dia_derm_acc, 4)}, fusion {round(test_dia_fusion_acc, 4)}')
        logger.info(f'Test PN Acc breakdown: mean {round(test_pn_acc, 4)}, clic {round(test_pn_clic_acc, 4)}, derm {round(test_pn_derm_acc, 4)}, fusion {round(test_pn_fusion_acc, 4)}')
        logger.info(f'Test STR Acc breakdown: mean {round(test_str_acc, 4)}, clic {round(test_str_clic_acc, 4)}, derm {round(test_str_derm_acc, 4)}, fusion {round(test_str_fusion_acc, 4)}')
        logger.info(f'Test PIG Acc breakdown: mean {round(test_pig_acc, 4)}, clic {round(test_pig_clic_acc, 4)}, derm {round(test_pig_derm_acc, 4)}, fusion {round(test_pig_fusion_acc, 4)}')
        logger.info(f'Test RS Acc breakdown: mean {round(test_rs_acc, 4)}, clic {round(test_rs_clic_acc, 4)}, derm {round(test_rs_derm_acc, 4)}, fusion {round(test_rs_fusion_acc, 4)}')
        logger.info(f'Test DAG Acc breakdown: mean {round(test_dag_acc, 4)}, clic {round(test_dag_clic_acc, 4)}, derm {round(test_dag_derm_acc, 4)}, fusion {round(test_dag_fusion_acc, 4)}')
        logger.info(f'Test BWV Acc breakdown: mean {round(test_bwv_acc, 4)}, clic {round(test_bwv_clic_acc, 4)}, derm {round(test_bwv_derm_acc, 4)}, fusion {round(test_bwv_fusion_acc, 4)}')
        logger.info(f'Test VS Acc breakdown: mean {round(test_vs_acc, 4)}, clic {round(test_vs_clic_acc, 4)}, derm {round(test_vs_derm_acc, 4)}, fusion {round(test_vs_fusion_acc, 4)}')
        logger.info(f'Thus, test SPS Acc breakdown: mean {round(test_sps_acc, 4)}, clic {round(test_sps_clic_acc, 4)}, derm {round(test_sps_derm_acc, 4)}, fusion {round(test_sps_fusion_acc, 4)}')
        logger.info("")
    else:
        test_time = 0
        msg = {
            'is_best_val': is_best_val,
            'test time': test_time,
        }   
    total_test_time += test_time
    send_msg(s1, msg)


    # communicating time
    send_msg(s1, {'server_to_client_communication_time': epoch_communication_time_server_to_client})
    rmsg = recv_msg(s1)
    epoch_communication_time_client_to_server = rmsg['client_to_server_communication_time']
    logger.info("")
    logger.info(f"Epoch: client to server com time: {round(epoch_communication_time_client_to_server, 2)}")
    logger.info(f"Epoch: server to client com. time: {round(epoch_communication_time_server_to_client, 2)}")
    total_communication_time_server_to_client += epoch_communication_time_server_to_client    
    total_communication_time_client_to_server += epoch_communication_time_client_to_server
    logger.info(f"Epoch: training time client: {round(epoch_training_time, 2)}")
    logger.info(f"Epoch: training time server: {round(epoch_server_training_time, 2)}")
    # logger.info(f"Epoch: training time server (over-approximation): {round(epoch_training_time_server, 2)}")
    logger.info(f"Epoch: validation time: {round(validation_time, 2)}")
    logger.info(f"Epoch: test time: {round(test_time, 2)}")
    logger.info(f"Eopch: server load time: {round(server_load_time, 2)}")
    logger.info(f"Epoch: compression and decompression time: {round(epoch_compresson_decompression_time, 2)}")
    epoch_end_time = time.time()
    logger.info(f"Epoch: total time: {round(epoch_end_time - epoch_start_time, 2)}")
    
    
    
    logger.info("")
    logger.info(f"Epoch: received msg len from server: {round((received_msg_len - epoch_received_msg_len)/1024/1024, 2)} MB")
    logger.info(f"Epoch: size of server output: {round(epoch_size_server_output/1024/1024, 2)} MB")
    total_size_server_output += epoch_size_server_output
    logger.info(f"Epoch: size of client first gradient: {round(epoch_size_client_first_gradient/1024/1024, 2)} MB")
    total_size_client_first_gradient += epoch_size_client_first_gradient
    logger.info(f"Epoch: server model size: {round(server_model_size/1024/1024, 2)} MB")



    # store result in a csv file
    entry['exp_seq'] = exp_seq
    entry['epoch'] = epc
    entry['total_epochs'] = epochs
    entry['learning_rate'] = lr
    entry['architecture_choice'] = args.architecture_choice

    entry['batch_size'] = batch_size
    entry['len_train_dataset'] = len(train_dataloader.dataset)
    entry['len_val_dataset'] = len(val_dataloader.dataset)
    entry['len_test_dataset'] = len(test_dataloader.dataset)

    entry['param_client_first'] = sum(p.numel() for p in client_model_first.parameters() if p.requires_grad)
    entry['param_server_middle'] = sum(p.numel() for p in server_model.parameters() if p.requires_grad)
    entry['param_client_last'] = sum(p.numel() for p in client_model_last.parameters() if p.requires_grad)
    
    entry['train_loss'] = train_loss
    entry['train_dia_acc'] = train_dia_acc
    entry['train_sps_acc'] = train_sps_acc
    entry['train_mean_acc'] = train_mean_acc
    entry['val_loss'] = val_loss
    entry['val_dia_acc'] = val_dia_acc
    entry['val_sps_acc'] = val_sps_acc
    entry['val_mean_acc'] = val_mean_acc
    entry['is_best_val'] = is_best_val

    # breakdown
    entry['val_dia_clic_acc'] = val_dia_clic_acc
    entry['val_dia_derm_acc'] = val_dia_derm_acc
    entry['val_dia_fusion_acc'] = val_dia_fusion_acc
    entry['val_sps_clic_acc'] = val_sps_clic_acc
    entry['val_sps_derm_acc'] = val_sps_derm_acc
    entry['val_sps_fusion_acc'] = val_sps_fusion_acc
    entry['val_pn_acc'] = val_pn_acc
    entry['val_str_acc'] = val_str_acc
    entry['val_pig_acc'] = val_pig_acc
    entry['val_rs_acc'] = val_rs_acc
    entry['val_dag_acc'] = val_dag_acc
    entry['val_bwv_acc'] = val_bwv_acc
    entry['val_vs_acc'] = val_vs_acc
    entry['val_pn_clic_acc'] = val_pn_clic_acc
    entry['val_pn_derm_acc'] = val_pn_derm_acc
    entry['val_pn_fusion_acc'] = val_pn_fusion_acc
    entry['val_str_clic_acc'] = val_str_clic_acc
    entry['val_str_derm_acc'] = val_str_derm_acc
    entry['val_str_fusion_acc'] = val_str_fusion_acc
    entry['val_pig_clic_acc'] = val_pig_clic_acc
    entry['val_pig_derm_acc'] = val_pig_derm_acc
    entry['val_pig_fusion_acc'] = val_pig_fusion_acc
    entry['val_rs_clic_acc'] = val_rs_clic_acc
    entry['val_rs_derm_acc'] = val_rs_derm_acc
    entry['val_rs_fusion_acc'] = val_rs_fusion_acc
    entry['val_dag_clic_acc'] = val_dag_clic_acc
    entry['val_dag_derm_acc'] = val_dag_derm_acc
    entry['val_dag_fusion_acc'] = val_dag_fusion_acc
    entry['val_bwv_clic_acc'] = val_bwv_clic_acc
    entry['val_bwv_derm_acc'] = val_bwv_derm_acc
    entry['val_bwv_fusion_acc'] = val_bwv_fusion_acc
    entry['val_vs_clic_acc'] = val_vs_clic_acc
    entry['val_vs_derm_acc'] = val_vs_derm_acc
    entry['val_vs_fusion_acc'] = val_vs_fusion_acc

    if(is_best_val):
        entry['test_loss'] = test_loss
        entry['test_dia_acc'] = test_dia_acc
        entry['test_sps_acc'] = test_sps_acc
        entry['test_mean_acc'] = test_mean_acc

        # breakdown
        entry['test_dia_clic_acc'] = test_dia_clic_acc
        entry['test_dia_derm_acc'] = test_dia_derm_acc
        entry['test_dia_fusion_acc'] = test_dia_fusion_acc
        entry['test_sps_clic_acc'] = test_sps_clic_acc
        entry['test_sps_derm_acc'] = test_sps_derm_acc
        entry['test_sps_fusion_acc'] = test_sps_fusion_acc
        entry['test_pn_acc'] = test_pn_acc
        entry['test_str_acc'] = test_str_acc
        entry['test_pig_acc'] = test_pig_acc
        entry['test_rs_acc'] = test_rs_acc
        entry['test_dag_acc'] = test_dag_acc
        entry['test_bwv_acc'] = test_bwv_acc
        entry['test_vs_acc'] = test_vs_acc
        entry['test_pn_clic_acc'] = test_pn_clic_acc
        entry['test_pn_derm_acc'] = test_pn_derm_acc
        entry['test_pn_fusion_acc'] = test_pn_fusion_acc
        entry['test_str_clic_acc'] = test_str_clic_acc
        entry['test_str_derm_acc'] = test_str_derm_acc
        entry['test_str_fusion_acc'] = test_str_fusion_acc
        entry['test_pig_clic_acc'] = test_pig_clic_acc
        entry['test_pig_derm_acc'] = test_pig_derm_acc
        entry['test_pig_fusion_acc'] = test_pig_fusion_acc
        entry['test_rs_clic_acc'] = test_rs_clic_acc
        entry['test_rs_derm_acc'] = test_rs_derm_acc
        entry['test_rs_fusion_acc'] = test_rs_fusion_acc
        entry['test_dag_clic_acc'] = test_dag_clic_acc
        entry['test_dag_derm_acc'] = test_dag_derm_acc
        entry['test_dag_fusion_acc'] = test_dag_fusion_acc
        entry['test_bwv_clic_acc'] = test_bwv_clic_acc
        entry['test_bwv_derm_acc'] = test_bwv_derm_acc
        entry['test_bwv_fusion_acc'] = test_bwv_fusion_acc
        entry['test_vs_clic_acc'] = test_vs_clic_acc
        entry['test_vs_derm_acc'] = test_vs_derm_acc
        entry['test_vs_fusion_acc'] = test_vs_fusion_acc


    else:
        entry['test_loss'] = None
        entry['test_dia_acc'] = None
        entry['test_sps_acc'] = None
        entry['test_mean_acc'] = None

        # breakdown
        entry['test_dia_clic_acc'] = None
        entry['test_dia_derm_acc'] = None
        entry['test_dia_fusion_acc'] = None
        entry['test_sps_clic_acc'] = None
        entry['test_sps_derm_acc'] = None
        entry['test_sps_fusion_acc'] = None
        entry['test_pn_acc'] = None
        entry['test_str_acc'] = None
        entry['test_pig_acc'] = None
        entry['test_rs_acc'] = None
        entry['test_dag_acc'] = None
        entry['test_bwv_acc'] = None
        entry['test_vs_acc'] = None
        entry['test_pn_clic_acc'] = None
        entry['test_pn_derm_acc'] = None
        entry['test_pn_fusion_acc'] = None
        entry['test_str_clic_acc'] = None
        entry['test_str_derm_acc'] = None
        entry['test_str_fusion_acc'] = None
        entry['test_pig_clic_acc'] = None
        entry['test_pig_derm_acc'] = None
        entry['test_pig_fusion_acc'] = None
        entry['test_rs_clic_acc'] = None
        entry['test_rs_derm_acc'] = None
        entry['test_rs_fusion_acc'] = None
        entry['test_dag_clic_acc'] = None
        entry['test_dag_derm_acc'] = None
        entry['test_dag_fusion_acc'] = None
        entry['test_bwv_clic_acc'] = None
        entry['test_bwv_derm_acc'] = None
        entry['test_bwv_fusion_acc'] = None
        entry['test_vs_clic_acc'] = None
        entry['test_vs_derm_acc'] = None
        entry['test_vs_fusion_acc'] = None
        

    entry['time_server_load'] = server_load_time
    entry['time_client_training'] = epoch_training_time
    entry['time_server_training'] = epoch_server_training_time
    entry['time_validation'] = validation_time
    entry['time_test'] = test_time
    entry['time_communication_server_to_client'] = epoch_communication_time_server_to_client
    entry['time_communication_client_to_server'] = epoch_communication_time_client_to_server
    entry['time_compression_decompression'] = epoch_compresson_decompression_time
    entry['time_total'] = epoch_end_time - epoch_start_time

    entry['size_server_to_client_msg'] = received_msg_len - epoch_received_msg_len
    entry['size_server_output'] = epoch_size_server_output
    entry['size_client_head_gradient'] = epoch_size_client_first_gradient
    entry['size_server_model'] = server_model_size

    rmsg = recv_msg(s1)
    for key in rmsg.keys():
        if(key == 'communication_time_stamp'):
            continue
        entry[key] = rmsg[key]

    # from pprint import pprint
    # pprint(entry)
    
    # store results
    result = pd.DataFrame(entry, index=[0])
    result.to_csv(f'{save_path}/result.csv', header=False, index=False, mode='a')


    epoch_received_msg_len = received_msg_len

    if(epc == epochs-1):
        logger.info(", ".join(["\'" + column + "\'" for column in result.columns.tolist()]))
     

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
logger.info(f"Test time: {round(total_test_time, 2)}")
logger.info(f"Total time: {round(end_time - start_time, 2)}")


logger.info("")
logger.info(f"Received msg len from server: {round(received_msg_len/1024/1024, 2)} MB")
logger.info(f"Total size of server output: {round(total_size_server_output/1024/1024, 2)} MB")
logger.info(f"Total size of client first gradient: {round(total_size_client_first_gradient/1024/1024, 2)} MB")
logger.info(f"Total size of server model: {round(total_size_server_model/1024/1024, 2)} MB")
# 



#
