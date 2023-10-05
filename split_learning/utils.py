import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

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