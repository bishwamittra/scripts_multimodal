

from time import time
import json
from data import \
    (
        DermDataset, 
        derm_data, 
        clinic_train, 
        clinic_validate, 
        clinic_test,
        dermoscopic_train,
        dermoscopic_validate,
        dermoscopic_test,
        label_train_diag,
        label_validate_diag,
        label_test_diag,
        label_train_crit,
        label_validate_crit,
        label_test_crit
    )   
from torchvision import transforms
import pandas as pd
import torch
import numpy as np
import random
from torchvision import models
from model import \
    (
        CNN,
        Concate,
        Discriminator,
        ReconstructionNet
    )

import torch.nn as nn
import torch.optim as optim
from itertools import chain

from utils import \
    (
        train_fun,
        test_fun,
        validate_fun,
        metric,
        get_average_acc,
        get_average_auc,
        get_confusion_matrix,
        get_specificity,
        set_path,
        get_logger,
        MultiFocalLoss,
        get_scheduler,

    )


import argparse

def argparser():
    parser = argparse.ArgumentParser(description='centralized sgd baseline')

    # default args - data set and model
    parser.add_argument('--device', type=str, default='cuda', help='choose from cuda, cpu, mps')
    parser.add_argument('--seed', type=int, default=42, help='set a seed for reproducability')
    parser.add_argument('--lr', type=float, default=1e-5, help='server learning rate for updating global model by the server')
    parser.add_argument('--batch_size', type=int, default=12, help='server batch size for training global model')
    parser.add_argument('--epoch', type=int, default=5, help='server epochs to train global model with synthetic data')
    parser.add_argument('--save_root', type=str, default='./results/', help='path to save results')
    # parser.add_argument('--one_gpu', action='store_true', default=False, help='use only one gpu')

    args = parser.parse_args()

    return args


def main(args, logger):



    # device = torch.device("cuda:0")# ("cuda:0")
    device = args.device



    derm_data.dataset_stats()


    # print(len(clinic_train), len(clinic_validate), len(clinic_test))
    # print(clinic_train)

    class_sample_count = np.array([len(np.where(label_train_diag == t)[0]) for t in np.unique(label_train_diag)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in label_train_diag])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    train_transforms = transforms.Compose([transforms.Resize([299, 299]),
                                        # transforms.Pad(padding=10, fill=(255, 176, 145)),
                                        transforms.RandomCrop([299, 299], padding=20, padding_mode='edge'),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation([-45, 45]),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5 , 0.5 , 0.5), (0.5 , 0.5 , 0.5))])
    test_transforms = transforms.Compose([transforms.Resize([299, 299]),
        transforms.ToTensor(),
        transforms.Normalize((0.5 , 0.5 , 0.5), (0.5 , 0.5 , 0.5))
    ])
    image_transforms = {'train':train_transforms, 'test':test_transforms}

    train = list(zip(clinic_train, dermoscopic_train, label_train_diag, label_train_crit))
    train_df = pd.DataFrame(train, columns=['c_path','d_path','lab_diag', 'lab_crit'])
    train_dataset = DermDataset(train_df, transform=image_transforms['train'])

    validate = list(zip(clinic_validate, dermoscopic_validate, label_validate_diag, label_validate_crit))
    validate_df = pd.DataFrame(validate, columns=['c_path','d_path','lab_diag', 'lab_crit'])
    validate_dataset = DermDataset(validate_df, transform=image_transforms['test'])

    test = list(zip(clinic_test, dermoscopic_test, label_test_diag, label_test_crit))
    test_df = pd.DataFrame(test, columns=['c_path','d_path','lab_diag', 'lab_crit'])
    test_dataset = DermDataset(test_df, transform=image_transforms['test'])


    resnet50 = models.resnet50(pretrained=True)
    resnet501 = models.resnet50(pretrained=True)
    cnn_c = CNN(resnet50).to(device)
    cnn_d = CNN(resnet501).to(device)
    concate_net = Concate().to(device)
    discriminator = Discriminator().to(device)# 判别分布

    
    reconstruct_net_c = ReconstructionNet(in_feature=2048 * 2, output_size=(299, 299)).to(device)
    reconstruct_net_d = ReconstructionNet(in_feature=2048 * 2, output_size=(299, 299)).to(device)



    learning_rate = 1e-5
    learning_rate_re = 1e-5
    criterion = nn.CrossEntropyLoss()
    criterion1 = MultiFocalLoss(num_class = 2, gamma=2)# nn.CrossEntropyLoss()
    criterion2 = MultiFocalLoss(num_class = 3, gamma=2)# nn.CrossEntropyLoss()
    criterion3 = MultiFocalLoss(num_class = 5, gamma=2)# nn.CrossEntropyLoss()
    opt_list = chain(cnn_c.parameters(), cnn_d.parameters(), concate_net.parameters(), reconstruct_net_c.parameters(), reconstruct_net_d.parameters(), discriminator.parameters())
    # optimizer = optim.Adam(chain(reconstruct_net_c.parameters(), reconstruct_net_d.parameters(), concate_net.parameters(), cnn_c.parameters(), cnn_d.parameters()), lr=learning_rate , weight_decay=0.0001) #
    optimizer = optim.AdamW(opt_list, lr=learning_rate, weight_decay=0.0001) # , weight_decay=0.0001
    # optimizer_con = optim.Adam(chain(concate_net.parameters(), discriminator.parameters()), lr=learning_rate) # , weight_decay=0.0001
    # optimizer_re = optim.Adam(chain(reconstruct_net_c.parameters(), reconstruct_net_d.parameters()), lr=learning_rate_re) # , weight_decay=0.0001
    # criterion_recon = nn.MSELoss()
    scheduler = get_scheduler(optimizer, 'warmstart')
    criterion_recon = nn.MSELoss(reduction='none')
    criterion_l1 = nn.L1Loss(reduction='none')


    epochs = 150
    record_acc = 0.
    record_auc = 0.

    record_acc1 = 0.
    record_auc1 = 0.

    log_file = open('./log/log' + 'concate_reconstruct_attention_fusion_new' + '.txt', 'w', buffering = 1)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

    model_name_c = './checkpoint/feature_extraction_c_fusion_new.pth' # 27-3 ahieved the best performance # 0502-2 record result # 0502-3 best results
    model_name_d = './checkpoint/feature_extraction_d_fusion_new.pth' #  1-8or9
    model_name_concate = './checkpoint/concatenate_fusion_new.pth'
    model_name_reconstruct_c = './checkpoint/reconstruct_c_fusion_new.pth'
    model_name_reconstruct_d = './checkpoint/reconstruct_d_fusion_new.pth'
    model_name_discriminator = './checkpoint/discriminator_fusion_new.pth'
    for i in range(epochs):


        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size= 8,
                                                    sampler=sampler, num_workers=4)
        validateloader = torch.utils.data.DataLoader(validate_dataset, batch_size=48,
                                                    shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=48,
                                                    shuffle=False, num_workers=4)
        print("Epoch {} begin training...".format(i))
        log_file.write("Epoch {} begin training...\n".format(i))
        pred_all_train, label_true_train = train_fun(trainloader, cnn_c, cnn_d, concate_net, reconstruct_net_c, reconstruct_net_d, i) 
        auc_all_train, acc_all_train, con_all_train = metric(pred_all_train, label_true_train, show=False) # auc_all, acc_all, con_all
        avg_acc_train = get_average_acc(acc_all_train)# get the average acc
        avg_auc_train = get_average_auc(auc_all_train)
        print(avg_acc_train, avg_auc_train)
        log_file.write("Current average ACC: {:.4f} \n".format(avg_acc_train))
        log_file.write("Current average AUC: {:.4f} \n".format(avg_auc_train))
    #     scheduler.step()
        
        print(scheduler.get_lr()[0])

        print("Epoch {} begin validating...".format(i))
        log_file.write("Epoch {} begin validating...\n".format(i))
        pred_all_validate, label_true_validate = validate_fun(validateloader, cnn_c, cnn_d, concate_net, reconstruct_net_c, reconstruct_net_d, i)
        auc_all_validate, acc_all_validate, con_all_validate = metric(pred_all_validate, label_true_validate, show=False)
        avg_acc_validate = get_average_acc(acc_all_validate)# get the average acc
        avg_auc_validate = get_average_auc(auc_all_validate)
        print(avg_acc_validate, avg_auc_validate)
        log_file.write("Current average ACC: {:.4f} \n".format(avg_acc_validate))
        log_file.write("Current average AUC: {:.4f} \n".format(avg_auc_validate))

        print("Epoch {} begin testing...".format(i))
        log_file.write("Epoch {} begin testing...\n".format(i))
        pred_all_test, label_true_test = test_fun(testloader, cnn_c, cnn_d, concate_net, reconstruct_net_c, reconstruct_net_d, i)
        auc_all_test, acc_all_test, con_all_test = metric(pred_all_test, label_true_test, show=False)
        avg_acc = get_average_acc(acc_all_test)# get the average acc
        avg_auc = get_average_auc(auc_all_test)
        con_metric = get_confusion_matrix(pred_all_test, label_true_test) # compute recall and precision
        specificity = get_specificity(pred_all_test, label_true_test)
        # sens, spec, prec = get_confusion_matrix(con_all_test)
        # if i % 10 == 0 or i == (epochs - 1):
        if (record_acc+record_auc) <= (avg_acc_validate + avg_auc_validate):
            record_acc1 = avg_acc_validate
            record_auc1 = avg_auc_validate
            print("Best validate test metics on epoch {}:".format(i))
            print(auc_all_test)
            print(acc_all_test)
            print(avg_acc)
            print(avg_auc)
            log_file.write("Best validate test metics on epoch {}:\n".format(i))
            log_file.write(str(auc_all_test) + '\n')
            log_file.write(str(acc_all_test) + '\n')
            log_file.write('confusion_matrix' + str(con_metric) + '\n')
            
            torch.save(cnn_c.state_dict(), model_name_c)
            torch.save(cnn_d.state_dict(), model_name_d)
            torch.save(concate_net.state_dict(), model_name_concate)
            torch.save(reconstruct_net_c.state_dict(), model_name_reconstruct_c)
            torch.save(reconstruct_net_d.state_dict(), model_name_reconstruct_d)
            torch.save(reconstruct_net_d.state_dict(), model_name_discriminator)
            
            print("Test metics on epoch {}:".format(i))
            print(auc_all_test)
            print(acc_all_test)
            print(avg_acc)
            print(avg_auc)
            log_file.write("Test metics on epoch {}:\n".format(i))
            log_file.write(str(auc_all_test) + '\n')
            log_file.write(str(acc_all_test) + '\n')
            log_file.write('confusion_matrix' + str(con_metric) + '\n')
            log_file.write(str(avg_acc))
            log_file.write(str(avg_auc))

    log_file.close()
            


    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    # validloader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)


    # record_acc = 0.
    # record_auc = 0.
    # total_time_train = 0
    # for i in range(args.epoch):
    #         # training
    #         start_time_epoch_train = time()

    #         # print("Epoch {} begin train...".format(i))
    #         logger.info("Epoch {} begin train...".format(i))
    #         pred_all_train, label_true_train = train_func(trainloader, 
    #                 cnn_c, 
    #                 cnn_d, 
    #                 concate_net, 
    #                 reconstruct_net_c, 
    #                 reconstruct_net_d, 
    #                 optimizer, 
    #                 criterion, 
    #                 device, 
    #                 i
    #             )
    #         auc_all_train, acc_all_train, con_all_train = metric(pred_all_train, label_true_train, show=False)
    #         avg_acc = get_average_acc(acc_all_train)# get the average acc
    #         avg_auc = get_average_auc(auc_all_train)
    #         con_metric = get_confusion_matrix(pred_all_train, label_true_train) # compute recall and precision
    #         specificity = get_specificity(pred_all_train, label_true_train)
    #         # sens, spec, prec = get_confusion_matrix(con_all_test)

    #         if (True):
    #             record_acc = avg_acc
    #             record_auc = avg_auc
    #             logger.info(f"auc all train: {auc_all_train}")
    #             logger.info(f"acc all train: {acc_all_train}")
    #             logger.info("train average ACC: {:.4f}".format(avg_acc))
    #             logger.info("train average AUC: {:.4f}".format(avg_auc))
                
    #         logger.info("")

    #         end_time_epoch_train = time()
    #         epoch_time_train = end_time_epoch_train - start_time_epoch_train
    #         total_time_train += epoch_time_train


    #         # validation
    #         logger.info("Epoch {} begin validating...".format(i))
    #         pred_all_valid, label_true_valid = test_func(validloader,
    #                                 cnn_c, 
    #                                 cnn_d, 
    #                                 concate_net, 
    #                                 reconstruct_net_c, 
    #                                 reconstruct_net_d, 
    #                                 criterion,
    #                                 device,
    #                                 i)
    #         auc_all_valid, acc_all_valid, con_all_valid = metric(pred_all_valid, label_true_valid, show=False)
    #         avg_acc = get_average_acc(acc_all_valid)# get the average acc
    #         avg_auc = get_average_auc(auc_all_valid)
    #         con_metric = get_confusion_matrix(pred_all_valid, label_true_valid) # compute recall and precision
    #         specificity = get_specificity(pred_all_valid, label_true_valid)
    #         # sens, spec, prec = get_confusion_matrix(con_all_test)
    #         # if i % 10 == 0 or i == (epochs - 1):
    #         if (record_acc+record_auc) <= (avg_acc + avg_auc):
    #             record_acc = avg_acc
    #             record_auc = avg_auc
    #             logger.info(f"auc all valid: {auc_all_valid}")
    #             logger.info(f"acc all valid: {acc_all_valid}")
    #             logger.info("Current best average ACC: {:.4f}".format(avg_acc))
    #             logger.info("Current average AUC: {:.4f}".format(avg_auc))

    #         logger.info("")

            




    # record_acc = 0.
    # record_auc = 0.

    # # checkpoints
    # # model_name_c = './checkpoint/feature_extraction_c_fusion_9-12_21.pth'# './checkpoint/feature_extraction_concate_discrinimator_0713_c1_two_stream.pth' # 3 ahieved the best performance 
    # # model_name_d = './checkpoint/feature_extraction_d_fusion_9-12_21.pth' #'./checkpoint/feature_extraction_concate_discrinimator_0713_d1_two_stream.pth'
    # # model_name_concate = './checkpoint/concatenate_fusion_9-12_21.pth'# './checkpoint/concate_discrinimator_0713_concatenate1_two_stream.pth'
    # # model_name_dis_c = './checkpoint/discriminator_fusion_9-12_21.pth'# './checkpoint/reconstruct_concate_discrinimator_0713_c1_two_stream.pth'
    # # model_name_recon_c = './checkpoint/reconstruct_c_fusion_9-12_21.pth'# './checkpoint/feature_extraction_concate_recon_0713_c1_two_stream.pth' # 3 ahieved the best performance
    # # model_name_recon_d = './checkpoint/reconstruct_c_fusion_9-12_21.pth'# './checkpoint/feature_extraction_concate_recon_0713_d1_two_stream.pth'

    # # checkpoint_c = torch.load(model_name_c)
    # # cnn_c.load_state_dict(checkpoint_c)
    # # checkpoint_d = torch.load(model_name_d)
    # # cnn_d.load_state_dict(checkpoint_d)
    # # checkpoint_concate_net = torch.load(model_name_concate)
    # # concate_net.load_state_dict(checkpoint_concate_net)

    # testloader = torch.utils.data.DataLoader(test_dataset, batch_size=48,
    #                                             shuffle=False, num_workers=8)

    # # print("\n\n\n===========================================\n\n\n")
    # logger.info("\n\n\n===========================================\n\n\n")
    # i=1
    # logger.info("Epoch {} begin testing...".format(i))
    # pred_all_test, label_true_test = test_func(testloader, 
    #                         cnn_c, 
    #                         cnn_d, 
    #                         concate_net, 
    #                         reconstruct_net_c, 
    #                         reconstruct_net_d, 
    #                         criterion,
    #                         device,
    #                         i)
    # auc_all_test, acc_all_test, con_all_test = metric(pred_all_test, label_true_test, show=False)
    # avg_acc = get_average_acc(acc_all_test)# get the average acc
    # avg_auc = get_average_auc(auc_all_test)
    # con_metric = get_confusion_matrix(pred_all_test, label_true_test) # compute recall and precision
    # specificity = get_specificity(pred_all_test, label_true_test)
    # # sens, spec, prec = get_confusion_matrix(con_all_test)
    # # if i % 10 == 0 or i == (epochs - 1):
    # if (record_acc+record_auc) <= (avg_acc + avg_auc):
    #     record_acc = avg_acc
    #     record_auc = avg_auc
    #     logger.info("Test metics on epoch {}:".format(i))
    #     logger.info(f"auc all test: {auc_all_test}")
    #     logger.info(f"acc all test: {acc_all_test}")
    #     logger.info("Current best average ACC: {:.4f}".format(avg_acc))
    #     logger.info("Current average AUC: {:.4f}".format(avg_auc))



    # Epoch 1 begin testing...
    # Epoch: 1 test loss, Diag loss: 1.2603, PN loss: 0.9970, STR loss: 0.7590, PIG loss: 0.8560, RS loss: 0.5059, DaG loss: 1.0333, BWV loss: 0.4149, VS loss: 0.5281
    # 0.6420886075949368 0.8651865120908606
    # Test metics on epoch 1:
    # {'diag': {0: 0.7076187335092349, 1: 0.6360782482357824, 2: 0.6217754428504074, 3: 0.5730281690140845, 4: 0.7340425531914894, 'micro': 0.8328168562730331, 'macro': 0.6561713272151666}, 'pn': {0: 0.8749061259521511, 1: 0.816333828464543, 2: 0.8398134301787368, 'micro': 0.7575484697965069, 'macro': 0.8453706595143892}, 'str': {0: 0.8349968984379406, 1: 0.8205128205128205, 2: 0.8478122570156217, 'micro': 0.8604999198846339, 'macro': 0.8366352876824144}, 'pig': {0: 0.9056731671707164, 1: 0.873859269932757, 2: 0.9168253779311987, 'micro': 0.8371767344976766, 'macro': 0.9008540747608589}, 'rs': {0: 0.8709277273617549, 1: 0.8709277273617549, 'micro': 0.8773017144688351, 'macro': 0.8732082053778082}, 'dag': {0: 0.8211525423728814, 1: 0.8106528789084011, 2: 0.8519670346757892, 'micro': 0.757224803717353, 'macro': 0.8297087122067603}, 'bwv': {0: 0.8657916666666666, 1: 0.8657916666666665, 'micro': 0.9226598301554237, 'macro': 0.8686601019965277}, 'vs': {0: 0.9313099041533548, 1: 0.9501569858712715, 2: 0.8691324200913243, 'micro': 0.9509277359397532, 'macro': 0.9201227966268323}}
    # {'diag': 0.5544303797468354, 'pn': 0.5721518987341773, 'str': 0.6506329113924051, 'pig': 0.5645569620253165, 'rs': 0.7316455696202532, 'dag': 0.4607594936708861, 'bwv': 0.810126582278481, 'vs': 0.7924050632911392}
    # Current best average ACC: 0.6421
    # Current average AUC: 0.8652


if __name__ == '__main__':

    # read arguments
    args = argparser()
    set_path(args)

    # save configuration json file
    with open(args.config_path, 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))
        f.close()

    # set the logger
    logger = get_logger(args.logger_path)

    # set seed for this experiment trial
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)     
    
    # set device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # run the main function to train model
    main(args, logger)

