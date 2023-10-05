import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import logging
import os

def get_metrics(net, eval_loader, device):
    net.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    
    with torch.no_grad():
        logits_all, targets_all = torch.tensor([], device='cpu'), torch.tensor([], dtype=torch.int, device='cpu')
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            logits_all = torch.cat((logits_all, logits.detach().cpu()),dim=0)
            targets_all = torch.cat((targets_all, y.cpu()), dim=0)
    
    pred = F.log_softmax(logits_all, dim=1)
    loss = criterion(pred, targets_all)/len(eval_loader.dataset) # validation loss
    
    output = pred.argmax(dim=1) # predicated/output label
    prob = F.softmax(logits_all, dim=1) # probabilities

    acc = accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
    bal_acc = balanced_accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
    auc = roc_auc_score(targets_all.numpy(), prob.numpy(), multi_class='ovr')

    return loss.item(), acc, auc, bal_acc



# generic purpose utils
def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

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


def set_path(save_root='result'):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    save_tag = f"split_learning"
    exp_seq_path = os.path.join(save_root, 'exp_seq.txt')

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

    exp_seq = exp_seq
    save_path = os.path.join(save_root, save_tag)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # config_path = os.path.join(save_path, 'config.json')
    logger_path = os.path.join(save_path, 'exp_log.log')    

    # server_save_path = os.path.join(save_path, 'server')
    # if not os.path.exists(server_save_path):
    #     os.makedirs(server_save_path)

    return logger_path, exp_seq


def get_logger(save_root='result'):

    logger_path, exp_seq = set_path(save_root=save_root)

    logging.basicConfig(
        filename=logger_path,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M', 
        level=logging.DEBUG, 
        filemode='w'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    return logger, exp_seq 