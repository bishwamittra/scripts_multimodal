
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
from utils import get_logger
import time
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from model_server import FusionNet
from dependency import class_list
import torch
import pickle
import struct
import socket
server_name = 'SERVER_001'


# Setup CUDA
seed_num = 777
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch.manual_seed(seed_num)
# if device == "cuda:0":
#     torch.cuda.manual_seed_all(seed_num)
# device = "cpu"
device = "cuda:0"
logger, exp_seq, save_path = get_logger(filename_prefix="server_")
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
    msg = recv_all(sock, msg_len)
    msg = pickle.loads(msg)
    global total_communication_time
    global offset_time
    total_communication_time += time.time() - \
        msg['communication_time_stamp'] + offset_time
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


server_model = FusionNet(class_list).to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = optim.Adam(server_model.parameters(), lr=lr)


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
rmsg, data_size = recv_msg(conn)
epochs = rmsg['epoch']
num_batch = rmsg['total_batch']
# setting the first communication time as 0 to offset the time.
offset_time = - total_communication_time


logger.info(f"received epoch: {rmsg['epoch']}, {rmsg['total_batch']}")

# send server meta information.
send_msg(conn, {"server_name": server_name, "server_time": time.time()})

# Start training
start_time = time.time()
logger.info(f"Start training @ {time.asctime()}")
server_model.set_mode('train')
for epc in range(epochs):
    train_loss = 0
    train_dia_acc = 0
    train_sps_acc = 0
    for index in tqdm(range(num_batch), ncols=100, desc='Training with {}'.format(server_name)):
        optimizer.zero_grad()

        # receives label and feature from client.
        msg, data_size = recv_msg(conn)

        # label
        label = msg['label']
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
        x_clic_cpu, x_derm_cpu = msg['x_clic'], msg['x_derm']
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
            logit_diagnosis_fusion, diagnosis_label), num_batch)
        dia_acc_clic = torch.true_divide(server_model.metric(
            logit_diagnosis_clic, diagnosis_label), num_batch)
        dia_acc_derm = torch.true_divide(server_model.metric(
            logit_diagnosis_derm, diagnosis_label), num_batch)

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
                                                7 * num_batch)

        # seven-point accuracy of clic
        sps_acc_clic = torch.true_divide(server_model.metric(logit_pn_clic, pn_label)
                                         + server_model.metric(logit_str_clic, str_label)
                                         + server_model.metric(logit_pig_clic, pig_label)
                                         + server_model.metric(logit_rs_clic, rs_label)
                                         + server_model.metric(logit_dag_clic, dag_label)
                                         + server_model.metric(logit_bwv_clic, bwv_label)
                                         + server_model.metric(logit_vs_clic, vs_label), 
                                                7 * num_batch)
        # seven-point accuracy of derm
        sps_acc_derm = torch.true_divide(server_model.metric(logit_pn_derm, pn_label)
                                         + server_model.metric(logit_str_derm, str_label)
                                         + server_model.metric(logit_pig_derm, pig_label)
                                         + server_model.metric(logit_rs_derm, rs_label)
                                         + server_model.metric(logit_dag_derm, dag_label)
                                         + server_model.metric(logit_bwv_derm, bwv_label)
                                         + server_model.metric(logit_vs_derm, vs_label), 
                                                7 * num_batch)
        # average seven-point accuracy
        sps_acc = torch.true_divide(sps_acc_fusion + sps_acc_clic + sps_acc_derm, 3)
        loss.backward()

        # send gradient to client
        msg = {
            "x_clic_grad": x_clic_cpu.grad.clone().detach(),
            "x_derm_grad": x_derm_cpu.grad.clone().detach(),
        }
        data_size = send_msg(conn, msg)
        optimizer.step()

        train_loss += loss.item()
        train_dia_acc += dia_acc.item()
        train_sps_acc += sps_acc.item()

    
    train_loss = train_loss / (index + 1)
    train_dia_acc = train_dia_acc / (index + 1)
    train_sps_acc = train_sps_acc / (index + 1)


    # logging
    logger.info(f"Round: ---, epoch: {epc+1}/{epochs}, Train Loss: {round(train_loss, 2)}, Train Dia Acc: {round(train_dia_acc, 2)}, Train SPS Acc: {round(train_sps_acc, 2)}")

        # if (i + 1) % 100 == 0:
        #     # measure accuracy and record loss
        #     _, predicted = torch.max(output, 1)
        #     correct = (predicted == label).sum().item()
        #     accuracy = correct / len(label)
        #     logger.info(f'Epoch: {epc+1}/{epoch}, Batch: {i+1}/{num_batch}, Train Loss: {round(loss.item(), 2)} Train Accuracy: {round(accuracy, 2)} Client to server communication time: {round(total_communication_time, 2)}')

        # if (i + 1) % 1000 == 0:
        #     logger.info("Start validation")
        #     # validation
        #     # receive total bach number and epoch from client.
        #     rmsg, data_size = recv_msg(conn)
        #     num_test_batch = rmsg['num_batch']
        #     test_dataset_size = rmsg['dataset_size']
        #     resnet_server.eval()
        #     with torch.no_grad():
        #         logits_all, targets_all = torch.tensor(
        #             [], device='cpu'), torch.tensor([], dtype=torch.int, device='cpu')
        #         # for j in range(num_test_batch):
        #         for j in range(num_test_batch):
        #             msg, data_size = recv_msg(conn)
        #             # label
        #             label = msg['label']
        #             # conversion between gpu and cpu.
        #             label = label.clone().detach().long().to(device)

        #             # feature
        #             client_output_cpu = msg['client_output']
        #             client_output = client_output_cpu.to(device)

        #             # forward propagation
        #             logits = resnet_server(client_output)
        #             logits_all = torch.cat(
        #                 (logits_all, logits.detach().cpu()), dim=0)
        #             targets_all = torch.cat((targets_all, label.cpu()), dim=0)

        #         pred = F.log_softmax(logits_all, dim=1)
        #         test_loss = criterion(pred, targets_all) / \
        #             test_dataset_size  # validation loss

        #         output = pred.argmax(dim=1)  # predicated/output label
        #         prob = F.softmax(logits_all, dim=1)  # probabilities

        #         test_acc = accuracy_score(
        #             y_pred=output.numpy(), y_true=targets_all.numpy())
        #         test_bal_acc = balanced_accuracy_score(
        #             y_pred=output.numpy(), y_true=targets_all.numpy())
        #         test_auc = roc_auc_score(
        #             targets_all.numpy(), prob.numpy(), multi_class='ovr')
        #         logger.info(
        #             f'Test Loss: {round(test_loss.item(), 2)} Test Accuracy: {round(test_acc, 2)} Test AUC: {round(test_auc, 2)} Test Balanced Accuracy: {round(test_bal_acc, 2)}')

        #     # break
        #     server_to_client_communication_time = recv_msg(
        #         conn)[0]['server_to_client_communication_time']
        #     logger.info(
        #         f"Server to client communication time: {server_to_client_communication_time}")

        #     # break

server_to_client_communication_time = recv_msg(
    conn)[0]['server_to_client_communication_time']

logger.info(f'Contribution from {server_name} is done')
logger.info(
    f"Client to server communication time: {round(total_communication_time, 2)}")
logger.info(
    f"Server to client communication time: {server_to_client_communication_time}")
logger.info(f'Total duration is: {round(time.time() - start_time, 2)} seconds')
