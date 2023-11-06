
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
from utils import get_logger
import time
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from model_server import FusionNet_server
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
args = parser.parse_args()



# Setup CUDA
device = "cuda:1"
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)     
logger, exp_seq, save_path = get_logger(filename_prefix="server_")
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


server_model = FusionNet_server(class_list).to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = optim.Adam(server_model.parameters(), lr=lr)


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
batch_size = rmsg['batch_size']
num_batch = rmsg['num_batch']
logger.info(f"received epoch: {rmsg['epoch']}, batch size: {rmsg['batch_size']}, num_batch: {rmsg['num_batch']}")


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
    epoch_training_time = 0
    epoch_size_client_head_output = 0
    epoch_size_server_gradient = 0
    

    train_loss = 0
    train_dia_acc = 0
    train_sps_acc = 0

    for index in tqdm(range(num_batch), ncols=100, desc='Training with {}'.format(server_name)):
        batch_training_start_time = time.time()
        optimizer.zero_grad()
        epoch_training_time += time.time() - batch_training_start_time

        size_client_head_output = received_msg_len
        rmsg = recv_msg(conn) # receives label and feature from client.
        size_client_head_output = received_msg_len - size_client_head_output
        epoch_size_client_head_output += size_client_head_output


        # forward propagation
        batch_training_start_time = time.time()
        # label
        label = rmsg['label']
        # Diagostic label
        diagnosis_label = label[0].clone().long().to(device)
        # Seven-Point Checklikst labels
        pn_label = label[1].clone().long().to(device)
        str_label = label[2].clone().long().to(device)
        pig_label = label[3].clone().long().to(device)
        rs_label = label[4].clone().long().to(device)
        dag_label = label[5].clone().long().to(device)
        bwv_label = label[6].clone().long().to(device)
        vs_label = label[7].clone().long().to(device)

        # feature
        x_clic_cpu, x_derm_cpu = rmsg['x_clic'], rmsg['x_derm']
        x_clic = x_clic_cpu.to(device)
        x_derm = x_derm_cpu.to(device)

        # forward propagation
        [(logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm,
          logit_vs_derm),
         (logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic, logit_bwv_clic,
          logit_vs_clic),
         (logit_diagnosis_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion,
          logit_vs_fusion)] = server_model((x_clic, x_derm))

        # average fusion loss
        loss_fusion = torch.true_divide(
            server_model.criterion(logit_diagnosis_fusion, diagnosis_label)
            + server_model.criterion(logit_pn_fusion, pn_label)
            + server_model.criterion(logit_str_fusion, str_label)
            + server_model.criterion(logit_pig_fusion, pig_label)
            + server_model.criterion(logit_rs_fusion, rs_label)
            + server_model.criterion(logit_dag_fusion, dag_label)
            + server_model.criterion(logit_bwv_fusion, bwv_label)
            + server_model.criterion(logit_vs_fusion, vs_label), 8)

        # average clic loss
        loss_clic = torch.true_divide(
            server_model.criterion(logit_diagnosis_clic, diagnosis_label)
            + server_model.criterion(logit_pn_clic, pn_label)
            + server_model.criterion(logit_str_clic, str_label)
            + server_model.criterion(logit_pig_clic, pig_label)
            + server_model.criterion(logit_rs_clic, rs_label)
            + server_model.criterion(logit_dag_clic, dag_label)
            + server_model.criterion(logit_bwv_clic, bwv_label)
            + server_model.criterion(logit_vs_clic, vs_label), 8)

        # average derm loss
        loss_derm = torch.true_divide(
            server_model.criterion(logit_diagnosis_derm, diagnosis_label)
            + server_model.criterion(logit_pn_derm, pn_label)
            + server_model.criterion(logit_str_derm, str_label)
            + server_model.criterion(logit_pig_derm, pig_label)
            + server_model.criterion(logit_rs_derm, rs_label)
            + server_model.criterion(logit_dag_derm, dag_label)
            + server_model.criterion(logit_bwv_derm, bwv_label)
            + server_model.criterion(logit_vs_derm, vs_label), 8)

        # average loss
        loss = loss_fusion*0.33 + loss_clic*0.33 + loss_derm*0.33

        # fusion, clic, derm accuracy for diagnostic
        dia_acc_fusion = torch.true_divide(server_model.metric(
            logit_diagnosis_fusion, diagnosis_label), batch_size)
        dia_acc_clic = torch.true_divide(server_model.metric(
            logit_diagnosis_clic, diagnosis_label), batch_size)
        dia_acc_derm = torch.true_divide(server_model.metric(
            logit_diagnosis_derm, diagnosis_label), batch_size)

        # average accuracy for diagnostic
        dia_acc = torch.true_divide(
            dia_acc_fusion + dia_acc_clic + dia_acc_derm, 3)

        # seven-point accuracy of fusion
        sps_acc_fusion = torch.true_divide(server_model.metric(logit_pn_fusion, pn_label)
                                           + server_model.metric(logit_str_fusion, str_label)
                                           + server_model.metric(logit_pig_fusion, pig_label)
                                           + server_model.metric(logit_rs_fusion, rs_label)
                                           + server_model.metric(logit_dag_fusion, dag_label)
                                           + server_model.metric(logit_bwv_fusion, bwv_label)
                                           + server_model.metric(logit_vs_fusion, vs_label), 
                                                7 * batch_size)

        # seven-point accuracy of clic
        sps_acc_clic = torch.true_divide(server_model.metric(logit_pn_clic, pn_label)
                                         + server_model.metric(logit_str_clic, str_label)
                                         + server_model.metric(logit_pig_clic, pig_label)
                                         + server_model.metric(logit_rs_clic, rs_label)
                                         + server_model.metric(logit_dag_clic, dag_label)
                                         + server_model.metric(logit_bwv_clic, bwv_label)
                                         + server_model.metric(logit_vs_clic, vs_label), 
                                                7 * batch_size)
        # seven-point accuracy of derm
        sps_acc_derm = torch.true_divide(server_model.metric(logit_pn_derm, pn_label)
                                         + server_model.metric(logit_str_derm, str_label)
                                         + server_model.metric(logit_pig_derm, pig_label)
                                         + server_model.metric(logit_rs_derm, rs_label)
                                         + server_model.metric(logit_dag_derm, dag_label)
                                         + server_model.metric(logit_bwv_derm, bwv_label)
                                         + server_model.metric(logit_vs_derm, vs_label), 
                                                7 * batch_size)
        # average seven-point accuracy
        sps_acc = torch.true_divide(sps_acc_fusion + sps_acc_clic + sps_acc_derm, 3)
        train_loss += loss.item()
        train_dia_acc += dia_acc.item()
        train_sps_acc += sps_acc.item()
        loss.backward()


        # send gradient to client
        msg = {
            "x_clic_grad": x_clic_cpu.grad.clone().detach(),
            "x_derm_grad": x_derm_cpu.grad.clone().detach(),
        }
        epoch_training_time += time.time() - batch_training_start_time

        send_msg(conn, msg)
        
        batch_training_start_time = time.time()
        optimizer.step()
        epoch_training_time += time.time() - batch_training_start_time

        # break

    train_loss = train_loss / (index + 1)
    train_dia_acc = train_dia_acc / (index + 1)
    train_sps_acc = train_sps_acc / (index + 1)
    total_training_time += epoch_training_time

    # for validation and test, send server model to client
    send_msg(conn, {"server model": {k: v.cpu() for k, v in server_model.state_dict().items()}}) # send model to client.
    

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
    logger.info(f"Test loss: {round(rmsg['test loss'], 4)}, Test dia acc: {round(rmsg['test dia acc'], 4)}, Test sps acc: {round(rmsg['test sps acc'], 4)}, Test mean acc: {round(rmsg['test mean acc'], 4)}")
    test_time = rmsg['test time']
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
    logger.info(f"Epoch: total time: {round(time.time() - epoch_start_time, 2)}")
    total_communication_time_client_to_server += epoch_communication_time_client_to_server


    logger.info("")
    logger.info(f"Epoch: received msg len from client: {round((received_msg_len - epoch_received_msg_len)/1024/1024, 2)} MB")
    logger.info(f"Epoch: size of client head output: {round(epoch_size_client_head_output/1024/1024, 2)} MB")
    # logger.info(f"Epoch: size of server gradient: {round(epoch_size_server_gradient/1024/1024, 2)} MB")
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
logger.info(f'Total duration is: {round(time.time() - start_time, 2)} seconds')
logger.info("")
logger.info(f"Received msg len from client: {round(received_msg_len/1024/1024, 2)} MB")
logger.info(f"Total size of client head output: {round(total_size_client_head_output/1024/1024, 2)} MB")
# logger.info(f"Total size of server gradient: {round(total_size_server_gradient/1024/1024, 2)} MB")

    


    