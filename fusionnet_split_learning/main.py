from utils import get_logger, validation, train
import torch.optim as optim
from _model import FusionNet
import numpy as np
import pandas as pd
import torch
import time
import random
import os
import os
from dependency import *
from dataloader import generate_dataloader
import argparse

    


parser = argparse.ArgumentParser()
parser.add_argument('--connection_start_from_client', action='store_true', default=False)
parser.add_argument('--epoch', type=int, default=2, help='Number of epochs')
parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda', choices=['cpu', 'cuda'])
parser.add_argument('--cuda_id', type=int, default=1, help='cuda id')
parser.add_argument('--architecture_choice', type=int, default=0, help='Index of architecture choice')
parser.add_argument('--client_in_sambanova', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()


logger, exp_seq, save_path = get_logger(filename_prefix="fusion_net_non_split")
logger.info(f"-------------------------Session: Exp {exp_seq}")

assert args.architecture_choice == 0
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)     



epochs = args.epoch
# lr = 3e-5
lr = 0.001
batch_size = 32
num_workers = 8
if(args.device == 'cuda'):
    cuda_id = args.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_id}"
    device = f"cuda:{cuda_id}"
else:
    device = args.device
shape = (224, 224)
model = FusionNet(class_list).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)



train_dataloader, val_dataloader, test_dataloader = generate_dataloader(shape, batch_size, num_workers)

logger.info("Start training...")

best_mean_acc = 0
# best_loss = np.inf
is_best_val = False

for epoch in range(epochs):

    epoch_start_time = time.time()


    train_loss, \
    train_dia_acc, \
    [train_dia_acc_clic, train_dia_acc_derm, train_dia_acc_fusion], \
    train_sps_acc, \
    [train_sps_acc_clic, train_sps_acc_derm, train_sps_acc_fusion], \
    [train_pn_acc, train_str_acc, train_pig_acc, train_rs_acc, train_dag_acc, train_bwv_acc, train_vs_acc], \
    [
        [train_pn_clic_acc, train_pn_derm_acc, train_pn_fusion_acc], 
        [train_str_clic_acc, train_str_derm_acc, train_str_fusion_acc], 
        [train_pig_clic_acc, train_pig_derm_acc, train_pig_fusion_acc], 
        [train_rs_clic_acc, train_rs_derm_acc, train_rs_fusion_acc], 
        [train_dag_clic_acc, train_dag_derm_acc, train_dag_fusion_acc], 
        [train_bwv_clic_acc, train_bwv_derm_acc, train_bwv_fusion_acc], 
        [train_vs_clic_acc, train_vs_derm_acc, train_vs_fusion_acc]
    ] = train(model, train_dataloader, optimizer, device)
    train_mean_acc = (train_dia_acc*1 + train_sps_acc*7)/8
    epoch_training_time = time.time() - epoch_start_time


    
    # logger.info(f"Round: ---, epoch: {epoch}, Train Loss: {round(train_loss, 2)}, Train Dia Acc: {round(train_dia_acc, 2)}, Train SPS Acc: {round(train_sps_acc, 2)}")
    logger.info("")
    logger.info(f"Epoch {epoch+1}/{epochs} results:")
    logger.info(f'Train Loss: {round(train_loss, 4)}, Train Dia Acc: {round(train_dia_acc, 4)}, Train SPS Acc: {round(train_sps_acc, 4)} Train Mean Acc: {round(train_mean_acc, 4)}')
    # breakdown accuracy
    logger.info(f'Train Dia Acc breakdown: mean {round(train_dia_acc, 4)} clic {round(train_dia_acc_clic, 4)}, derm {round(train_dia_acc_derm, 4)}, fusion {round(train_dia_acc_fusion, 4)}')
    logger.info(f'Train PN Acc breakdown: mean {round(train_pn_acc, 4)}, clic {round(train_pn_clic_acc, 4)}, derm {round(train_pn_derm_acc, 4)}, fusion {round(train_pn_fusion_acc, 4)}')
    logger.info(f'Train STR Acc breakdown: mean {round(train_str_acc, 4)}, clic {round(train_str_clic_acc, 4)}, derm {round(train_str_derm_acc, 4)}, fusion {round(train_str_fusion_acc, 4)}')
    logger.info(f'Train PIG Acc breakdown: mean {round(train_pig_acc, 4)}, clic {round(train_pig_clic_acc, 4)}, derm {round(train_pig_derm_acc, 4)}, fusion {round(train_pig_fusion_acc, 4)}')
    logger.info(f'Train RS Acc breakdown: mean {round(train_rs_acc, 4)}, clic {round(train_rs_clic_acc, 4)}, derm {round(train_rs_derm_acc, 4)}, fusion {round(train_rs_fusion_acc, 4)}')
    logger.info(f'Train DAG Acc breakdown: mean {round(train_dag_acc, 4)}, clic {round(train_dag_clic_acc, 4)}, derm {round(train_dag_derm_acc, 4)}, fusion {round(train_dag_fusion_acc, 4)}')
    logger.info(f'Train BWV Acc breakdown: mean {round(train_bwv_acc, 4)}, clic {round(train_bwv_clic_acc, 4)}, derm {round(train_bwv_derm_acc, 4)}, fusion {round(train_bwv_fusion_acc, 4)}')
    logger.info(f'Train VS Acc breakdown: mean {round(train_vs_acc, 4)}, clic {round(train_vs_clic_acc, 4)}, derm {round(train_vs_derm_acc, 4)}, fusion {round(train_vs_fusion_acc, 4)}')
    logger.info(f'Thus, valid SPS Acc breakdown: mean {round(train_sps_acc, 4)}, clic {round(train_sps_acc_clic, 4)}, derm {round(train_sps_acc_derm, 4)}, fusion {round(train_sps_acc_fusion, 4)}')    
    logger.info("")



    # validation mode
    epoch_validation_start_time = time.time()
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
    ] = validation(model, val_dataloader, device)
    val_mean_acc = (val_dia_acc*1 + val_sps_acc*7)/8
    epoch_validation_time = time.time() - epoch_validation_start_time

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
        is_best_val = True
        best_mean_acc = val_mean_acc
        # torch.save(model.state_dict(), f'{save_path}/checkpoint/fusionnet_first_stage.pth')
        logger.info(f'Current Best Mean Validation Acc is {round(best_mean_acc, 2)}')
    
        # test mode
        epoch_test_start_time = time.time()
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
        ] = validation(model, test_dataloader, device)
        test_mean_acc = (test_dia_acc*1 + test_sps_acc*7)/8
        epoch_test_time = time.time() - epoch_test_start_time
        logger.info("")
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
        

    else:
        epoch_test_time = 0
        

    logger.info("")


    # store result in a csv file
    entry = {}
    entry['exp_seq'] = exp_seq
    entry['epoch'] = epoch
    entry['total_epochs'] = epochs
    entry['learning_rate'] = lr
    entry['architecture_choice'] = 0

    entry['batch_size'] = batch_size
    entry['len_train_dataset'] = len(train_dataloader.dataset)
    entry['len_val_dataset'] = len(val_dataloader.dataset)
    entry['len_test_dataset'] = len(test_dataloader.dataset)

    entry['param_client_first'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    entry['param_server_middle'] = None
    entry['param_client_last'] = None
    
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
        

    entry['time_server_load'] = None
    entry['time_client_training'] = epoch_training_time
    entry['time_server_training'] = None
    entry['time_validation'] = epoch_validation_time
    entry['time_test'] = epoch_test_time
    entry['time_communication_server_to_client'] = None
    entry['time_communication_client_to_server'] = None
    entry['time_total'] = time.time() - epoch_start_time

    entry['size_server_to_client_msg'] = None
    entry['size_server_output'] = None
    entry['size_client_head_gradient'] = None
    entry['size_server_model'] = None
    entry['size_client_to_server_msg'] = None
    entry['size_client_head_output'] = None
    entry['size_server_gradient'] = None

    
    # from pprint import pprint
    # pprint(entry)
    
    # store results
    result = pd.DataFrame(entry, index=[0])
    result.to_csv(f'{save_path}/result.csv', header=False, index=False, mode='a')


    if(epoch == epochs-1):
        logger.info(", ".join(["\'" + column + "\'" for column in result.columns.tolist()]))




