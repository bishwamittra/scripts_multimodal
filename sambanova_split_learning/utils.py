import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def get_metrics_u_shaped(resnet_server, resnet_client_first, resnet_client_last, test_loader, criterion, device):
    resnet_server.eval()
    resnet_client_first.eval()
    resnet_client_last.eval()
    test_dataset_size = len(test_loader.dataset)
    # print(test_dataset_size)
    
    with torch.no_grad():
        logits_all, targets_all = torch.tensor([], device=device), torch.tensor([], dtype=torch.int, device=device)
        for x, label in tqdm(test_loader):
            x = x.to(device)
            label = label.to(device)
            client_output_head = resnet_client_first(x)
            server_output = resnet_server(client_output_head)
            logits = resnet_client_last(server_output)

            logits_all = torch.cat((logits_all, logits.detach().cpu()),dim=0)
            targets_all = torch.cat((targets_all, label.cpu()), dim=0)

            # print(criterion(logits, label).item())
            # break
            
        pred = F.log_softmax(logits_all, dim=1)
        test_loss = criterion(pred, targets_all)/test_dataset_size # validation loss
        
        output = pred.argmax(dim=1) # predicated/output label
        prob = F.softmax(logits_all, dim=1) # probabilities

        test_acc = accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
        test_bal_acc = balanced_accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
        test_auc = roc_auc_score(targets_all.numpy(), prob.numpy(), multi_class='ovr')

    return test_loss.item(), test_acc, test_auc, test_bal_acc



def get_metrics_u_shaped_client_side(resnet_client_first, 
                                     resnet_client_last, 
                                     test_loader, 
                                     send_msg,
                                     recv_msg,
                                     conn,
                                     criterion, 
                                     device):
    resnet_client_first.eval()
    resnet_client_last.eval()
    test_dataset_size = len(test_loader.dataset)
    # print(test_dataset_size)
    
    send_msg(conn, {'total_test_batch': len(test_loader)})

    with torch.no_grad():
        logits_all, targets_all = torch.tensor([], device=device), torch.tensor([], dtype=torch.int, device=device)
        for x, label in test_loader:
            x = x.to(device)
            label = label.to(device)
            client_output_first = resnet_client_first(x)
            client_output_first = client_output_first.clone().detach().requires_grad_(True)
            client_output_first = client_output_first.to(torch.bfloat16)

            msg = {'client_output': client_output_first}
            send_msg(conn, msg)
            rmsg = recv_msg(conn)
            server_output_samba = rmsg['server_output']
            server_output = server_output_samba.to(device)
            
            logits = resnet_client_last(server_output)

            logits_all = torch.cat((logits_all, logits.detach().cpu()),dim=0)
            targets_all = torch.cat((targets_all, label.cpu()), dim=0)

            
        pred = F.log_softmax(logits_all, dim=1)
        test_loss = criterion(pred, targets_all)/test_dataset_size # validation loss
        
        output = pred.argmax(dim=1) # predicated/output label
        prob = F.softmax(logits_all, dim=1) # probabilities

        test_acc = accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
        test_bal_acc = balanced_accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
        test_auc = roc_auc_score(targets_all.numpy(), prob.numpy(), multi_class='ovr')

    return test_loss.item(), test_acc, test_auc, test_bal_acc




