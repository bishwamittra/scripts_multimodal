from utils import get_logger, validation, train
import torch.optim as optim
from model import FusionNet
import numpy as np
import torch
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from dependency import *
from dataloader import generate_dataloader


def run_exp(model, epochs, optimizer, device, train_dataloader, val_dataloader, test_dataloader, logger, save_path):
    logger.info("Start training...")

    best_mean_acc = 0
    best_loss = np.inf

    for epoch in range(epochs):

        # train_mode
        train_loss, train_dia_acc, train_sps_acc = train(model, train_dataloader, optimizer, device)
        logger.info(f"Round: ---, epoch: {epoch}, Train Loss: {round(train_loss, 2)}, Train Dia Acc: {round(train_dia_acc, 2)}, Train SPS Acc: {round(train_sps_acc, 2)}")

        # validation mode
        val_loss, val_dia_acc, val_sps_acc = validation(model, val_dataloader, device)
        # val_acc = (val_dia_acc + val_sps_acc) / 2
        val_mean_acc = (val_dia_acc*1 + val_sps_acc*7)/8
        logger.info(f'Round: ---, epoch: {epoch}, Valid Loss: {round(val_loss, 2)}, Valid Dia Acc: {round(val_dia_acc, 2)}, Valid SPS Acc: {round(val_sps_acc, 2)}')

        # save the best model
        if val_mean_acc > best_mean_acc:
            best_mean_acc = val_mean_acc
            torch.save(model.state_dict(), f'{save_path}/checkpoint/fusionnet_first_stage.pth')
            logger.info(f'Current Best Mean Validation Acc is {round(best_mean_acc, 2)}')
        
        # test mode
        test_loss, test_dia_acc, test_sps_acc = validation(model, test_dataloader, device)
        logger.info(f'Round: ---, epoch: {epoch}, Test Loss: {round(test_loss, 2)}, Test Dia Acc: {round(test_dia_acc, 2)}, Test SPS Acc: {round(test_sps_acc, 2)}')
        

        logger.info("")


def main():
    logger, exp_seq, save_path = get_logger(filename_prefix="fusionNet")
    logger.info(f"-------------------------Session: Exp {exp_seq}")

    epochs = 200
    lr = 3e-5
    batch_size = 32
    num_workers = 8
    device = "cuda:0"
    shape = (224, 224)
    model = FusionNet(class_list).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    train_dataloader, val_dataloader, test_dataloader = generate_dataloader(shape, batch_size, num_workers)
    run_exp(model, epochs, optimizer, device, train_dataloader, val_dataloader, test_dataloader, logger, save_path)

if __name__ == '__main__':
    main()

