import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
import sys, os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from itertools import cycle
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from sklearn.metrics import classification_report
from imblearn.metrics import specificity_score
import logging
import warnings
warnings.filterwarnings('ignore')


def get_logger(logger_path):
    logging.basicConfig(
        filename=logger_path,
        # filename='/home/qinbin/test.log',
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M', 
        level=logging.DEBUG, 
        filemode='w'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    return logger 

def set_path(args):
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # prepare the save path
    save_tag = f'multimodal_seven_point_dataset_seed_{args.seed}-ep{args.epoch}-bs{args.batch_size}-lr{args.lr}' 

    # if args.save_results or args.save_curves:
    exp_seq_path = os.path.join(args.save_root, 'exp_seq.txt')
    if not os.path.exists(exp_seq_path):
        file = open(exp_seq_path, 'w')
        exp_seq=0
        exp_seq = str(exp_seq)
        file.write(exp_seq)
        file.close
        save_tag = 'exp_' + exp_seq + '_' + save_tag
    else:
        file = open(exp_seq_path, 'r')
        exp_seq = int(file.read())
        exp_seq += 1
        exp_seq = str(exp_seq)
        save_tag = 'exp_' + exp_seq + '_' + save_tag
        file = open(exp_seq_path, 'w')
        file.write(exp_seq)
        file.close()
    args.exp_seq = exp_seq
    args.save_path = os.path.join(args.save_root, save_tag)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.config_path = os.path.join(args.save_path, 'config.json')
    args.logger_path = os.path.join(args.save_path, 'exp_log.log')   
   
    return args



def cal_auc(pre, true, show = False):
    auc_all = {}
    
    plt.rc('font',family='Times New Roman')
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10,}
    for key in pre.keys():
        n_classes = np.array(pre[key]).shape[-1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        one_hot = torch.zeros(len(true[key]), n_classes).scatter_(1, torch.tensor(np.array(true[key]).reshape(len(np.array(true[key])), 1)), 1)
        for i in range(n_classes):
            # fpr[i], tpr[i], _ = roc_curve(one_hot[:, i], np.array(pre[key])[:, i])
            fpr[i], tpr[i], _ = roc_curve(one_hot[:, i], np.array(nn.Softmax(dim=1)(torch.Tensor(pre[key])))[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(np.array(one_hot).ravel(), np.array(nn.Softmax(dim=1)(torch.Tensor(pre[key]))).ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        if show == True and key =='diag':
            # Plot all ROC curves
            color_list = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'brown']
            lw=2
            plt.figure()

            colors = cycle(color_list[0:n_classes])
            
            name_diag=['BCC','NEV','MEL','MISC','SK']
            
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.1f})'
                         ''.format(name_diag[i], roc_auc[i] * 100))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right", prop=font1)
            plt.savefig('./visualization/ROC.pdf')
            plt.show()
        auc_all[key] = roc_auc
    return auc_all

def cal_con_matrix(pre, true):
    acc_all = {}
    con_all = {}
    tp_fp = {}
    for key in pre.keys():
        acc = accuracy_score(np.array(true[key]), np.argmax(np.array(pre[key]), axis=-1))
        con = confusion_matrix(np.array(true[key]), np.argmax(np.array(pre[key]), axis=-1))
        acc_all[key] = acc
        con_all[key] = con
    return acc_all, con_all

def metric(pre, true, show = False):
    auc_all = cal_auc(pre=pre, true=true, show = show)
    acc_all, con_all = cal_con_matrix(pre=pre, true=true)
    return auc_all, acc_all, con_all



def generate_label(batch_size):
    l_c = torch.zeros(batch_size)
    l_d = torch.ones(batch_size)
    label = torch.cat((l_c, l_d), dim=0)
    return label

def test_func(dataloader, model_c, model_d, model_concate, model_recon_c, model_recon_d, criterion, device, epoch):
    model_c.eval()
    model_d.eval()
    model_concate.eval()
    loss_diag_count = []
    loss_crit_pn_count = []
    loss_crit_str_count = []
    loss_crit_pig_count = []
    loss_crit_rs_count = []
    loss_crit_dag_count = []
    loss_crit_bwv_count = []
    loss_crit_vs_count = []
    label_true = {'diag':[], 'pn':[], 'str':[], 'pig':[], 'rs':[], 'dag':[], 'bwv':[], 'vs':[]}
    pred_all = {'diag':[], 'pn':[], 'str':[], 'pig':[], 'rs':[], 'dag':[], 'bwv':[], 'vs':[]}
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            img_c, img_d, lab_diag, lab_crit = data
            label_true['diag'].extend(lab_diag)
            img_c, img_d, lab_diag = img_c.to(device), img_d.to(device), lab_diag.to(device)
            # print(img_c.shape)
            # import pdb;pdb.set_trace()
            label_true['pn'].extend(lab_crit[:, 0]), \
                        label_true['str'].extend(lab_crit[:, 1]), label_true['pig'].extend(lab_crit[:, 2]), label_true['rs'].extend(lab_crit[:, 3]), \
                        label_true['dag'].extend(lab_crit[:, 4]), label_true['bwv'].extend(lab_crit[:, 5]), label_true['vs'].extend(lab_crit[:, 6])
            lab_crit_pn, lab_crit_str, lab_crit_pig, lab_crit_rs, lab_crit_dag, lab_crit_bwv, lab_crit_vs = lab_crit[:, 0].to(device), lab_crit[:, 1].to(device), lab_crit[:, 2].to(device), lab_crit[:, 3].to(device), lab_crit[:, 4].to(device), lab_crit[:, 5].to(device), lab_crit[:, 6].to(device)

            # print(lab_crit_dag)
            feature_c = model_c(img_c)# predict for each class
            feature_d = model_d(img_d)# predict for each class
            prediction = model_concate(feature_c, feature_d)
            out_diag, out_crit_pn, out_crit_str, out_crit_pig, out_crit_rs, out_crit_dag, out_crit_bwv, out_crit_vs, \
                        att_mask_c, att_mask_d, att_feature = prediction
            
            recon_pred_c = model_recon_c(att_feature)
            recon_pred_d = model_recon_d(att_feature)

            pred_all['diag'].extend(out_diag.cpu().detach().numpy()), pred_all['pn'].extend(out_crit_pn.cpu().detach().numpy()), \
                            pred_all['str'].extend(out_crit_str.cpu().detach().numpy()), pred_all['pig'].extend(out_crit_pig.cpu().detach().numpy()), pred_all['rs'].extend(out_crit_rs.cpu().detach().numpy()), \
                            pred_all['dag'].extend(out_crit_dag.cpu().detach().numpy()), pred_all['bwv'].extend(out_crit_bwv.cpu().detach().numpy()), pred_all['vs'].extend(out_crit_vs.cpu().detach().numpy())
            loss_diag = criterion(out_diag, lab_diag)
            loss_crit_pn = criterion(out_crit_pn, lab_crit_pn)
            loss_crit_str = criterion(out_crit_str, lab_crit_str)
            loss_crit_pig = criterion(out_crit_pig, lab_crit_pig)
            loss_crit_rs = criterion(out_crit_rs, lab_crit_rs)
            loss_crit_dag = criterion(out_crit_dag, lab_crit_dag)
            loss_crit_bwv = criterion(out_crit_bwv, lab_crit_bwv)
            loss_crit_vs = criterion(out_crit_vs, lab_crit_vs)
            loss_diag_count.append(loss_diag.item())
            loss_crit_pn_count.append(loss_crit_pn.item())
            loss_crit_str_count.append(loss_crit_str.item())
            loss_crit_pig_count.append(loss_crit_pig.item())
            loss_crit_rs_count.append(loss_crit_rs.item())
            loss_crit_dag_count.append(loss_crit_dag.item())
            loss_crit_bwv_count.append(loss_crit_bwv.item())
            loss_crit_vs_count.append(loss_crit_vs.item())
        print("Epoch: {} test loss, Diag loss: {:.4f}, PN loss: {:.4f}, STR loss: {:.4f}, PIG loss: {:.4f}, RS loss: {:.4f}, DaG loss: {:.4f}, BWV loss: {:.4f}, VS loss: {:.4f}".format(
                        epoch, np.average(loss_diag_count), np.average(loss_crit_pn_count), np.average(loss_crit_str_count), np.average(loss_crit_pig_count), np.average(loss_crit_rs_count), np.average(loss_crit_dag_count), np.average(loss_crit_bwv_count), np.average(loss_crit_vs_count)))
    return pred_all, label_true


def get_average_acc(acc):
    accs = []
    for key in acc.keys():
        accs.append(acc[key])
    avg_acc = np.average(accs)
    return avg_acc
def get_average_auc(auc):
    aucs = []
    for key in auc.keys():
        if key == 'diag':
            continue
        else:
            for key_i in auc[key].keys():
                if key_i == 'micro' or key_i == 'macro':
                    continue
                else:
                    aucs.append(auc[key][key_i])
    # print(len(aucs))
    avg_auc = np.average(aucs)
    return avg_auc
def get_specificity(pre, true):
    sen = {}
    for key in pre:
        sen[key] = specificity_score(np.array(pre[key]).argmax(axis=1), true[key], average=None)
    return sen
def get_confusion_matrix(pre, true): # recall and precision
    confusion_metric = {}
    for key in pre:
        # import pdb;pdb.set_trace()
        confusion_metric[key] = classification_report(np.array(pre[key]).argmax(axis=1), true[key], zero_division  = 1, output_dict=True)
    return confusion_metric



def train_func(dataloader, model_c, model_d, model_concate, model_recon_c, model_recon_d, optimizer, criterion, device, epoch):
    model_c.train()
    model_d.train()
    model_concate.train()
    loss_diag_count = []
    loss_crit_pn_count = []
    loss_crit_str_count = []
    loss_crit_pig_count = []
    loss_crit_rs_count = []
    loss_crit_dag_count = []
    loss_crit_bwv_count = []
    loss_crit_vs_count = []
    label_true = {'diag':[], 'pn':[], 'str':[], 'pig':[], 'rs':[], 'dag':[], 'bwv':[], 'vs':[]}
    pred_all = {'diag':[], 'pn':[], 'str':[], 'pig':[], 'rs':[], 'dag':[], 'bwv':[], 'vs':[]}
    for index, data in enumerate(dataloader):
        img_c, img_d, lab_diag, lab_crit = data
        label_true['diag'].extend(lab_diag)
        img_c, img_d, lab_diag = img_c.to(device), img_d.to(device), lab_diag.to(device)
        # print(img_c.shape)
        # import pdb;pdb.set_trace()
        label_true['pn'].extend(lab_crit[:, 0]), \
                    label_true['str'].extend(lab_crit[:, 1]), label_true['pig'].extend(lab_crit[:, 2]), label_true['rs'].extend(lab_crit[:, 3]), \
                    label_true['dag'].extend(lab_crit[:, 4]), label_true['bwv'].extend(lab_crit[:, 5]), label_true['vs'].extend(lab_crit[:, 6])
        lab_crit_pn, lab_crit_str, lab_crit_pig, lab_crit_rs, lab_crit_dag, lab_crit_bwv, lab_crit_vs = lab_crit[:, 0].to(device), lab_crit[:, 1].to(device), lab_crit[:, 2].to(device), lab_crit[:, 3].to(device), lab_crit[:, 4].to(device), lab_crit[:, 5].to(device), lab_crit[:, 6].to(device)

        optimizer.zero_grad()

        # print(lab_crit_dag)
        feature_c = model_c(img_c)# predict for each class
        feature_d = model_d(img_d)# predict for each class
        prediction = model_concate(feature_c, feature_d)
        out_diag, out_crit_pn, out_crit_str, out_crit_pig, out_crit_rs, out_crit_dag, out_crit_bwv, out_crit_vs, \
                    att_mask_c, att_mask_d, att_feature = prediction
        
        recon_pred_c = model_recon_c(att_feature)
        recon_pred_d = model_recon_d(att_feature)

        pred_all['diag'].extend(out_diag.cpu().detach().numpy()), pred_all['pn'].extend(out_crit_pn.cpu().detach().numpy()), \
                        pred_all['str'].extend(out_crit_str.cpu().detach().numpy()), pred_all['pig'].extend(out_crit_pig.cpu().detach().numpy()), pred_all['rs'].extend(out_crit_rs.cpu().detach().numpy()), \
                        pred_all['dag'].extend(out_crit_dag.cpu().detach().numpy()), pred_all['bwv'].extend(out_crit_bwv.cpu().detach().numpy()), pred_all['vs'].extend(out_crit_vs.cpu().detach().numpy())
        loss_diag = criterion(out_diag, lab_diag)
        loss_crit_pn = criterion(out_crit_pn, lab_crit_pn)
        loss_crit_str = criterion(out_crit_str, lab_crit_str)
        loss_crit_pig = criterion(out_crit_pig, lab_crit_pig)
        loss_crit_rs = criterion(out_crit_rs, lab_crit_rs)
        loss_crit_dag = criterion(out_crit_dag, lab_crit_dag)
        loss_crit_bwv = criterion(out_crit_bwv, lab_crit_bwv)
        loss_crit_vs = criterion(out_crit_vs, lab_crit_vs)
        loss_diag_count.append(loss_diag.item())
        loss_crit_pn_count.append(loss_crit_pn.item())
        loss_crit_str_count.append(loss_crit_str.item())
        loss_crit_pig_count.append(loss_crit_pig.item())
        loss_crit_rs_count.append(loss_crit_rs.item())
        loss_crit_dag_count.append(loss_crit_dag.item())
        loss_crit_bwv_count.append(loss_crit_bwv.item())
        loss_crit_vs_count.append(loss_crit_vs.item())

        loss = loss_diag + loss_crit_pn + loss_crit_str + loss_crit_pig + loss_crit_rs + loss_crit_dag + loss_crit_bwv + loss_crit_vs
        loss.backward()
        optimizer.step()

    print("Epoch: {} train loss, Diag loss: {:.4f}, PN loss: {:.4f}, STR loss: {:.4f}, PIG loss: {:.4f}, RS loss: {:.4f}, DaG loss: {:.4f}, BWV loss: {:.4f}, VS loss: {:.4f}".format(
                    epoch, np.average(loss_diag_count), np.average(loss_crit_pn_count), np.average(loss_crit_str_count), np.average(loss_crit_pig_count), np.average(loss_crit_rs_count), np.average(loss_crit_dag_count), np.average(loss_crit_bwv_count), np.average(loss_crit_vs_count)))
    return pred_all, label_true
