from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import torch.nn.functional as F
from dependency import *
import torch
import torch.nn as nn
import torchvision
import os



sigmoid = nn.Sigmoid()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class FusionNet_client_first(nn.Module):

    def __init__(self, architecture_choice):
        super(FusionNet_client_first, self).__init__()

        self.architecture_choice = architecture_choice

        # from pretrained resnet50
        if(self.architecture_choice <= 4):
            model_clinic = torchvision.models.resnet50(pretrained=True)
            model_derm = torchvision.models.resnet50(pretrained=True)
        else:
            model_clinic = torchvision.models.resnet101(pretrained=True)
            model_derm = torchvision.models.resnet101(pretrained=True)
            self.architecture_choice -= 5
        

        if(self.architecture_choice in [1, 2, 3, 4]):
            # self.num_label = class_list[0]
            # self.num_pn = class_list[1]
            # self.num_str = class_list[2]
            # self.num_pig = class_list[3]
            # self.num_rs = class_list[4]
            # self.num_dag = class_list[5]
            # self.num_bwv = class_list[6]
            # self.num_vs = class_list[7]
            # self.dropout = nn.Dropout(0.3)

            
            # define the clinic model
            self.conv1_cli = model_clinic.conv1
            self.bn1_cli = model_clinic.bn1
            self.relu_cli = model_clinic.relu
            self.maxpool_cli = model_clinic.maxpool
            if(self.architecture_choice in  [2, 4]):
                self.layer1_cli = model_clinic.layer1
            
    
            # self.layer2_cli = model_clinic.layer2
            # self.layer3_cli = model_clinic.layer3
            # self.layer4_cli = model_clinic.layer4
            # self.avgpool_cli = model_clinic.avgpool
            # # self.avgpool_cli = nn.MaxPool2d(7, 7)

            self.conv1_derm = model_derm.conv1
            self.bn1_derm = model_derm.bn1
            self.relu_derm = model_derm.relu
            self.maxpool_derm = model_derm.maxpool
            if(self.architecture_choice in  [2, 4]):
                self.layer1_derm = model_derm.layer1
            
            # self.layer2_derm = model_derm.layer2
            # self.layer3_derm = model_derm.layer3
            # self.layer4_derm = model_derm.layer4
            # self.avgpool_derm = model_derm.avgpool
            # # self.avgpool_derm = nn.MaxPool2d(7, 7)
            # # self.fc = self.model.fc

            
            # self.fc_fusion_ = nn.Sequential(
            #     nn.Linear(2048, 512),
            #     nn.BatchNorm1d(512),
            #     Swish_Module(),
            #     nn.Dropout(p=0.3),
            #     nn.Linear(512, 128),
            #     nn.BatchNorm1d(128),
            #     Swish_Module(),
            # )

            # self.derm_mlp = nn.Sequential(
            #     nn.Linear(2048, 512),
            #     nn.BatchNorm1d(512),
            #     Swish_Module(),
            #     nn.Dropout(p=0.3),
            #     nn.Linear(512, 128),
            #     nn.BatchNorm1d(128),
            #     Swish_Module(),
            # )

            # self.clin_mlp = nn.Sequential(
            #     nn.Linear(2048, 512),
            #     nn.BatchNorm1d(512),
            #     Swish_Module(),
            #     nn.Dropout(p=0.3),
            #     nn.Linear(512, 128),
            #     nn.BatchNorm1d(128),
            #     Swish_Module(),
            # )

            # self.fc_cli = nn.Linear(128, self.num_label)
            # self.fc_pn_cli = nn.Linear(128, self.num_pn)
            # self.fc_str_cli = nn.Linear(128, self.num_str)
            # self.fc_pig_cli = nn.Linear(128, self.num_pig)
            # self.fc_rs_cli = nn.Linear(128, self.num_rs)
            # self.fc_dag_cli = nn.Linear(128, self.num_dag)
            # self.fc_bwv_cli = nn.Linear(128, self.num_bwv)
            # self.fc_vs_cli = nn.Linear(128, self.num_vs)

            # # self.fc_derm_ = nn.Linear(2048, 512)
            # self.fc_derm = nn.Linear(128, self.num_label)
            # self.fc_pn_derm = nn.Linear(128, self.num_pn)
            # self.fc_str_derm = nn.Linear(128, self.num_str)
            # self.fc_pig_derm = nn.Linear(128, self.num_pig)
            # self.fc_rs_derm = nn.Linear(128, self.num_rs)
            # self.fc_dag_derm = nn.Linear(128, self.num_dag)
            # self.fc_bwv_derm = nn.Linear(128, self.num_bwv)
            # self.fc_vs_derm = nn.Linear(128, self.num_vs)
            # # self.fc_ft = nn.

            # self.fc_fusion = nn.Linear(128, self.num_label)
            # self.fc_pn_fusion = nn.Linear(128, self.num_pn)
            # self.fc_str_fusion = nn.Linear(128, self.num_str)
            # self.fc_pig_fusion = nn.Linear(128, self.num_pig)
            # self.fc_rs_fusion = nn.Linear(128, self.num_rs)
            # self.fc_dag_fusion = nn.Linear(128, self.num_dag)
            # self.fc_bwv_fusion = nn.Linear(128, self.num_bwv)
            # self.fc_vs_fusion = nn.Linear(128, self.num_vs)

    def forward(self, x):
        (x_clic, x_derm) = x

        if(self.architecture_choice in [1, 2, 3, 4]):
            # passed through the clinic model (pretained resnet50)
            x_clic = self.conv1_cli(x_clic)
            x_clic = self.bn1_cli(x_clic)
            x_clic = self.relu_cli(x_clic)
            x_clic = self.maxpool_cli(x_clic)
            if(self.architecture_choice in  [2, 4]):
                x_clic = self.layer1_cli(x_clic)
            # x_clic = self.layer2_cli(x_clic)
            # x_clic = self.layer3_cli(x_clic)
            # x_clic = self.layer4_cli(x_clic)
            # x_clic = self.avgpool_cli(x_clic)
            # x_clic = x_clic.view(x_clic.size(0), -1)

            # passed through the derm model (pretained resnet50)
            x_derm = self.conv1_derm(x_derm)
            x_derm = self.bn1_derm(x_derm)
            x_derm = self.relu_derm(x_derm)
            x_derm = self.maxpool_derm(x_derm)
            if(self.architecture_choice in  [2, 4]):
                x_derm = self.layer1_derm(x_derm)
            # x_derm = self.layer2_derm(x_derm)
            # x_derm = self.layer3_derm(x_derm)
            # x_derm = self.layer4_derm(x_derm)
            # x_derm = self.avgpool_derm(x_derm)
            # x_derm = x_derm.view(x_derm.size(0), -1)

            # # fusion of the two model outputs
            # x_fusion = torch.add(x_clic, x_derm)
            # x_fusion = self.fc_fusion_(x_fusion)
            # x_fusion = self.dropout(x_fusion)

            # # mlp ahead of clinic model
            # x_clic = self.clin_mlp(x_clic)
            # x_clic = self.dropout(x_clic)


            # # mlp ahead of derm model
            # x_derm = self.derm_mlp(x_derm)
            # x_derm = self.dropout(x_derm)


            # # logits of clinic mlp
            # logit_clic = self.fc_cli(x_clic)
            # logit_pn_clic = self.fc_pn_cli(x_clic)
            # logit_str_clic = self.fc_str_cli(x_clic)
            # logit_pig_clic = self.fc_pig_cli(x_clic)
            # logit_rs_clic = self.fc_rs_cli(x_clic)
            # logit_dag_clic = self.fc_dag_cli(x_clic)
            # logit_bwv_clic = self.fc_bwv_cli(x_clic)
            # logit_vs_clic = self.fc_vs_cli(x_clic)

            
            # # logits of derm mlp
            # logit_derm = self.fc_derm(x_derm)
            # logit_pn_derm = self.fc_pn_derm(x_derm)
            # logit_str_derm = self.fc_str_derm(x_derm)
            # logit_pig_derm = self.fc_pig_derm(x_derm)
            # logit_rs_derm = self.fc_rs_derm(x_derm)
            # logit_dag_derm = self.fc_dag_derm(x_derm)
            # logit_bwv_derm = self.fc_bwv_derm(x_derm)
            # logit_vs_derm = self.fc_vs_derm(x_derm)

            # # logits of fusion mlp
            # logit_fusion = self.fc_fusion(x_fusion)
            # logit_pn_fusion = self.fc_pn_fusion(x_fusion)
            # logit_str_fusion = self.fc_str_fusion(x_fusion)
            # logit_pig_fusion = self.fc_pig_fusion(x_fusion)
            # logit_rs_fusion = self.fc_rs_fusion(x_fusion)
            # logit_dag_fusion = self.fc_dag_fusion(x_fusion)
            # logit_bwv_fusion = self.fc_bwv_fusion(x_fusion)
            # logit_vs_fusion = self.fc_vs_fusion(x_fusion)

            # return [(logit_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm, logit_vs_derm),
            #         (logit_clic, logit_pn_clic, logit_str_clic, logit_pig_clic,
            #          logit_rs_clic, logit_dag_clic, logit_bwv_clic, logit_vs_clic),
            #         (logit_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion, logit_vs_fusion)]

        return (x_clic, x_derm)

    def criterion(self, logit, truth):
        loss = nn.CrossEntropyLoss()(logit, truth)
        return loss

    def criterion1(self, logit, truth):
        loss = nn.L1Loss()(logit, truth)
        return loss

    def metric(self, logit, truth):
        # prob = F.sigmoid(logit)
        _, prediction = torch.max(logit.data, 1)
        acc = torch.sum(prediction == truth)
        # acc = accuracy_score(y_pred=prediction.numpy(), y_true=truth.numpy())
        auc = roc_auc_score(truth.numpy(), prob.numpy(), multi_class='ovr')
        # bal_acc = balanced_accuracy_score(y_pred=prediction.numpy(), y_true=truth.numpy())
        # return acc, auc, bal_acc
        return acc

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError




class FusionNet_client_last(nn.Module):

    def __init__(self, class_list, architecture_choice):
        super(FusionNet_client_last, self).__init__()

        self.architecture_choice = architecture_choice

        # from pretrained resnet50
        if(self.architecture_choice > 4):
            self.architecture_choice -= 5

        if(self.architecture_choice in [1, 2, 3, 4]):
            self.architecture_choice = architecture_choice
            self.num_label = class_list[0]
            self.num_pn = class_list[1]
            self.num_str = class_list[2]
            self.num_pig = class_list[3]
            self.num_rs = class_list[4]
            self.num_dag = class_list[5]
            self.num_bwv = class_list[6]
            self.num_vs = class_list[7]
            self.dropout = nn.Dropout(0.3)

            # from pretrained resnet50
            # model_clinic = torchvision.models.resnet50(pretrained=True)
            # model_derm = torchvision.models.resnet50(pretrained=True)

            # define the clinic model
            # self.conv1_cli = model_clinic.conv1
            # self.bn1_cli = model_clinic.bn1
            # self.relu_cli = model_clinic.relu
            # self.maxpool_cli = model_clinic.maxpool
            # self.layer1_cli = model_clinic.layer1
            # self.layer2_cli = model_clinic.layer2
            # self.layer3_cli = model_clinic.layer3
            # self.layer4_cli = model_clinic.layer4
            # self.avgpool_cli = model_clinic.avgpool
            # # self.avgpool_cli = nn.MaxPool2d(7, 7)

            # self.conv1_derm = model_derm.conv1
            # self.bn1_derm = model_derm.bn1
            # self.relu_derm = model_derm.relu
            # self.maxpool_derm = model_derm.maxpool
            # self.layer1_derm = model_derm.layer1
            # self.layer2_derm = model_derm.layer2
            # self.layer3_derm = model_derm.layer3
            # self.layer4_derm = model_derm.layer4
            # self.avgpool_derm = model_derm.avgpool
            # # self.avgpool_derm = nn.MaxPool2d(7, 7)
            # # self.fc = self.model.fc

            if(self.architecture_choice in [3, 4]):
                self.fc_fusion_ = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.BatchNorm1d(512),
                    Swish_Module(),
                    nn.Dropout(p=0.3),
                    nn.Linear(512, 128),
                    nn.BatchNorm1d(128),
                    Swish_Module(),
                )

                self.derm_mlp = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.BatchNorm1d(512),
                    Swish_Module(),
                    nn.Dropout(p=0.3),
                    nn.Linear(512, 128),
                    nn.BatchNorm1d(128),
                    Swish_Module(),
                )

                self.clin_mlp = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.BatchNorm1d(512),
                    Swish_Module(),
                    nn.Dropout(p=0.3),
                    nn.Linear(512, 128),
                    nn.BatchNorm1d(128),
                    Swish_Module(),
                )

            self.fc_cli = nn.Linear(128, self.num_label)
            self.fc_pn_cli = nn.Linear(128, self.num_pn)
            self.fc_str_cli = nn.Linear(128, self.num_str)
            self.fc_pig_cli = nn.Linear(128, self.num_pig)
            self.fc_rs_cli = nn.Linear(128, self.num_rs)
            self.fc_dag_cli = nn.Linear(128, self.num_dag)
            self.fc_bwv_cli = nn.Linear(128, self.num_bwv)
            self.fc_vs_cli = nn.Linear(128, self.num_vs)

            # self.fc_derm_ = nn.Linear(2048, 512)
            self.fc_derm = nn.Linear(128, self.num_label)
            self.fc_pn_derm = nn.Linear(128, self.num_pn)
            self.fc_str_derm = nn.Linear(128, self.num_str)
            self.fc_pig_derm = nn.Linear(128, self.num_pig)
            self.fc_rs_derm = nn.Linear(128, self.num_rs)
            self.fc_dag_derm = nn.Linear(128, self.num_dag)
            self.fc_bwv_derm = nn.Linear(128, self.num_bwv)
            self.fc_vs_derm = nn.Linear(128, self.num_vs)
            # self.fc_ft = nn.

            self.fc_fusion = nn.Linear(128, self.num_label)
            self.fc_pn_fusion = nn.Linear(128, self.num_pn)
            self.fc_str_fusion = nn.Linear(128, self.num_str)
            self.fc_pig_fusion = nn.Linear(128, self.num_pig)
            self.fc_rs_fusion = nn.Linear(128, self.num_rs)
            self.fc_dag_fusion = nn.Linear(128, self.num_dag)
            self.fc_bwv_fusion = nn.Linear(128, self.num_bwv)
            self.fc_vs_fusion = nn.Linear(128, self.num_vs)

    def forward(self, x):
        (x_clic, x_derm, x_fusion) = x

        if(self.architecture_choice in [1, 2, 3, 4]):

            # passed through the clinic model (pretained resnet50)
            # x_clic = self.conv1_cli(x_clic)
            # x_clic = self.bn1_cli(x_clic)
            # x_clic = self.relu_cli(x_clic)
            # x_clic = self.maxpool_cli(x_clic)
            # x_clic = self.layer1_cli(x_clic)
            # x_clic = self.layer2_cli(x_clic)
            # x_clic = self.layer3_cli(x_clic)
            # x_clic = self.layer4_cli(x_clic)
            # x_clic = self.avgpool_cli(x_clic)
            # x_clic = x_clic.view(x_clic.size(0), -1)

            # passed through the derm model (pretained resnet50)
            # x_derm = self.conv1_derm(x_derm)
            # x_derm = self.bn1_derm(x_derm)
            # x_derm = self.relu_derm(x_derm)
            # x_derm = self.maxpool_derm(x_derm)
            # x_derm = self.layer1_derm(x_derm)
            # x_derm = self.layer2_derm(x_derm)
            # x_derm = self.layer3_derm(x_derm)
            # x_derm = self.layer4_derm(x_derm)
            # x_derm = self.avgpool_derm(x_derm)
            # x_derm = x_derm.view(x_derm.size(0), -1)

            # # fusion of the two model outputs
            # x_fusion = torch.add(x_clic, x_derm)
            if(self.architecture_choice in [3, 4]):
                x_fusion = self.fc_fusion_(x_fusion)
                x_fusion = self.dropout(x_fusion)

                # mlp ahead of clinic model
                x_clic = self.clin_mlp(x_clic)
                x_clic = self.dropout(x_clic)

                # mlp ahead of derm model
                x_derm = self.derm_mlp(x_derm)
                x_derm = self.dropout(x_derm)


            # logits of clinic mlp
            logit_clic = self.fc_cli(x_clic)
            logit_pn_clic = self.fc_pn_cli(x_clic)
            logit_str_clic = self.fc_str_cli(x_clic)
            logit_pig_clic = self.fc_pig_cli(x_clic)
            logit_rs_clic = self.fc_rs_cli(x_clic)
            logit_dag_clic = self.fc_dag_cli(x_clic)
            logit_bwv_clic = self.fc_bwv_cli(x_clic)
            logit_vs_clic = self.fc_vs_cli(x_clic)

            
            # logits of derm mlp
            logit_derm = self.fc_derm(x_derm)
            logit_pn_derm = self.fc_pn_derm(x_derm)
            logit_str_derm = self.fc_str_derm(x_derm)
            logit_pig_derm = self.fc_pig_derm(x_derm)
            logit_rs_derm = self.fc_rs_derm(x_derm)
            logit_dag_derm = self.fc_dag_derm(x_derm)
            logit_bwv_derm = self.fc_bwv_derm(x_derm)
            logit_vs_derm = self.fc_vs_derm(x_derm)

            # logits of fusion mlp
            logit_fusion = self.fc_fusion(x_fusion)
            logit_pn_fusion = self.fc_pn_fusion(x_fusion)
            logit_str_fusion = self.fc_str_fusion(x_fusion)
            logit_pig_fusion = self.fc_pig_fusion(x_fusion)
            logit_rs_fusion = self.fc_rs_fusion(x_fusion)
            logit_dag_fusion = self.fc_dag_fusion(x_fusion)
            logit_bwv_fusion = self.fc_bwv_fusion(x_fusion)
            logit_vs_fusion = self.fc_vs_fusion(x_fusion)

            return [(logit_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm, logit_vs_derm),
                    (logit_clic, logit_pn_clic, logit_str_clic, logit_pig_clic,
                    logit_rs_clic, logit_dag_clic, logit_bwv_clic, logit_vs_clic),
                    (logit_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion, logit_vs_fusion)]


    def criterion(self, logit, truth):
        loss = nn.CrossEntropyLoss()(logit, truth)
        return loss

    def criterion1(self, logit, truth):
        loss = nn.L1Loss()(logit, truth)
        return loss

    def metric(self, logit, truth):
        # prob = F.sigmoid(logit)
        _, prediction = torch.max(logit.data, 1)
        acc = torch.sum(prediction == truth)
        # acc = accuracy_score(y_pred=prediction.numpy(), y_true=truth.numpy())
        # auc = roc_auc_score(truth.numpy(), prob.numpy(), multi_class='ovr')
        # bal_acc = balanced_accuracy_score(y_pred=prediction.numpy(), y_true=truth.numpy())
        # return acc, auc, bal_acc
        return acc

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError
        

    def forward_propagate_and_loss_compute(self, x, label, batch_size, device):


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


        [(logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm,
          logit_vs_derm),
         (logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic, logit_bwv_clic,
          logit_vs_clic),
         (logit_diagnosis_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion,
          logit_vs_fusion)] = self.forward(x)


        # average fusion loss
        loss_fusion = torch.true_divide(
            self.criterion(logit_diagnosis_fusion, diagnosis_label)
            + self.criterion(logit_pn_fusion, pn_label)
            + self.criterion(logit_str_fusion, str_label)
            + self.criterion(logit_pig_fusion, pig_label)
            + self.criterion(logit_rs_fusion, rs_label)
            + self.criterion(logit_dag_fusion, dag_label)
            + self.criterion(logit_bwv_fusion, bwv_label)
            + self.criterion(logit_vs_fusion, vs_label), 8)

        # average clic loss
        loss_clic = torch.true_divide(
            self.criterion(logit_diagnosis_clic, diagnosis_label)
            + self.criterion(logit_pn_clic, pn_label)
            + self.criterion(logit_str_clic, str_label)
            + self.criterion(logit_pig_clic, pig_label)
            + self.criterion(logit_rs_clic, rs_label)
            + self.criterion(logit_dag_clic, dag_label)
            + self.criterion(logit_bwv_clic, bwv_label)
            + self.criterion(logit_vs_clic, vs_label), 8)

        # average derm loss
        loss_derm = torch.true_divide(
            self.criterion(logit_diagnosis_derm, diagnosis_label)
            + self.criterion(logit_pn_derm, pn_label)
            + self.criterion(logit_str_derm, str_label)
            + self.criterion(logit_pig_derm, pig_label)
            + self.criterion(logit_rs_derm, rs_label)
            + self.criterion(logit_dag_derm, dag_label)
            + self.criterion(logit_bwv_derm, bwv_label)
            + self.criterion(logit_vs_derm, vs_label), 8)

        # average loss
        loss = loss_fusion*0.33 + loss_clic*0.33 + loss_derm*0.33

        # fusion, clic, derm accuracy for diagnostic
        dia_fusion_acc = torch.true_divide(self.metric(
            logit_diagnosis_fusion, diagnosis_label), batch_size)
        dia_clic_acc = torch.true_divide(self.metric(
            logit_diagnosis_clic, diagnosis_label), batch_size)
        dia_derm_acc = torch.true_divide(self.metric(
            logit_diagnosis_derm, diagnosis_label), batch_size)

        # average accuracy for diagnostic
        dia_acc = torch.true_divide(
            dia_fusion_acc + dia_clic_acc + dia_derm_acc, 3)
        

        # disentangled accuracy for seven-point checklist
        pn_fusion_acc = torch.true_divide(self.metric(logit_pn_fusion, pn_label), batch_size)
        str_fusion_acc = torch.true_divide(self.metric(logit_str_fusion, str_label), batch_size)
        pig_fusion_acc = torch.true_divide(self.metric(logit_pig_fusion, pig_label), batch_size)
        rs_fusion_acc = torch.true_divide(self.metric(logit_rs_fusion, rs_label), batch_size)
        dag_fusion_acc = torch.true_divide(self.metric(logit_dag_fusion, dag_label), batch_size)
        bwv_fusion_acc = torch.true_divide(self.metric(logit_bwv_fusion, bwv_label), batch_size)
        vs_fusion_acc = torch.true_divide(self.metric(logit_vs_fusion, vs_label), batch_size)

        pn_clic_acc = torch.true_divide(self.metric(logit_pn_clic, pn_label), batch_size)
        str_clic_acc = torch.true_divide(self.metric(logit_str_clic, str_label), batch_size)
        pig_clic_acc = torch.true_divide(self.metric(logit_pig_clic, pig_label), batch_size)
        rs_clic_acc = torch.true_divide(self.metric(logit_rs_clic, rs_label), batch_size)
        dag_clic_acc = torch.true_divide(self.metric(logit_dag_clic, dag_label), batch_size)
        bwv_clic_acc = torch.true_divide(self.metric(logit_bwv_clic, bwv_label), batch_size)
        vs_clic_acc = torch.true_divide(self.metric(logit_vs_clic, vs_label), batch_size)

        pn_derm_acc = torch.true_divide(self.metric(logit_pn_derm, pn_label), batch_size)
        str_derm_acc = torch.true_divide(self.metric(logit_str_derm, str_label), batch_size)
        pig_derm_acc = torch.true_divide(self.metric(logit_pig_derm, pig_label), batch_size)
        rs_derm_acc = torch.true_divide(self.metric(logit_rs_derm, rs_label), batch_size)
        dag_derm_acc = torch.true_divide(self.metric(logit_dag_derm, dag_label), batch_size)
        bwv_derm_acc = torch.true_divide(self.metric(logit_bwv_derm, bwv_label), batch_size)
        vs_derm_acc = torch.true_divide(self.metric(logit_vs_derm, vs_label), batch_size)


        # average of each seven-point checklist accuracy
        pn_acc = torch.true_divide(pn_fusion_acc + pn_clic_acc + pn_derm_acc, 3)
        str_acc = torch.true_divide(str_fusion_acc + str_clic_acc + str_derm_acc, 3)
        pig_acc = torch.true_divide(pig_fusion_acc + pig_clic_acc + pig_derm_acc, 3)
        rs_acc = torch.true_divide(rs_fusion_acc + rs_clic_acc + rs_derm_acc, 3)
        dag_acc = torch.true_divide(dag_fusion_acc + dag_clic_acc + dag_derm_acc, 3)
        bwv_acc = torch.true_divide(bwv_fusion_acc + bwv_clic_acc + bwv_derm_acc, 3)
        vs_acc = torch.true_divide(vs_fusion_acc + vs_clic_acc + vs_derm_acc, 3)

        # print(f"pn: {pn_acc}, str: {str_acc}, pig: {pig_acc}, rs: {rs_acc}, dag: {dag_acc}, bwv: {bwv_acc}, vs: {vs_acc}")


        # average seven-point accuracy
        sps_fusion_acc = torch.true_divide(pn_fusion_acc + str_fusion_acc + pig_fusion_acc + rs_fusion_acc + dag_fusion_acc + bwv_fusion_acc + vs_fusion_acc, 7)
        sps_clic_acc = torch.true_divide(pn_clic_acc + str_clic_acc + pig_clic_acc + rs_clic_acc + dag_clic_acc + bwv_clic_acc + vs_clic_acc, 7)
        sps_derm_acc = torch.true_divide(pn_derm_acc + str_derm_acc + pig_derm_acc + rs_derm_acc + dag_derm_acc + bwv_derm_acc + vs_derm_acc, 7)

        

        # average seven-point accuracy by fusion, clic, derm
        sps_acc = torch.true_divide(sps_fusion_acc + sps_clic_acc + sps_derm_acc, 3)


        # print(f"fusion: {sps_fusion_acc}, clic: {sps_clic_acc}, derm: {sps_derm_acc}")
        # print(f"sps_acc: {sps_acc}")

        # print(torch.true_divide(pn_acc + str_acc + pig_acc + rs_acc + dag_acc + bwv_acc + vs_acc, 7), sps_fusion_acc)


        # # seven-point accuracy of fusion
        # sps_fusion_acc = torch.true_divide(self.metric(logit_pn_fusion, pn_label)
        #                                    + self.metric(logit_str_fusion, str_label)
        #                                    + self.metric(logit_pig_fusion, pig_label)
        #                                    + self.metric(logit_rs_fusion, rs_label)
        #                                    + self.metric(logit_dag_fusion, dag_label)
        #                                    + self.metric(logit_bwv_fusion, bwv_label)
        #                                    + self.metric(logit_vs_fusion, vs_label), 
        #                                         7 * batch_size)

        # # seven-point accuracy of clic
        # sps_clic_acc = torch.true_divide(self.metric(logit_pn_clic, pn_label)
        #                                  + self.metric(logit_str_clic, str_label)
        #                                  + self.metric(logit_pig_clic, pig_label)
        #                                  + self.metric(logit_rs_clic, rs_label)
        #                                  + self.metric(logit_dag_clic, dag_label)
        #                                  + self.metric(logit_bwv_clic, bwv_label)
        #                                  + self.metric(logit_vs_clic, vs_label), 
        #                                         7 * batch_size)
        # # seven-point accuracy of derm
        # sps_derm_acc = torch.true_divide(self.metric(logit_pn_derm, pn_label)
        #                                  + self.metric(logit_str_derm, str_label)
        #                                  + self.metric(logit_pig_derm, pig_label)
        #                                  + self.metric(logit_rs_derm, rs_label)
        #                                  + self.metric(logit_dag_derm, dag_label)
        #                                  + self.metric(logit_bwv_derm, bwv_label)
        #                                  + self.metric(logit_vs_derm, vs_label), 
        #                                         7 * batch_size)
        # # average seven-point accuracy
        # sps_acc = torch.true_divide(sps_fusion_acc + sps_clic_acc + sps_derm_acc, 3)



        return loss, \
                dia_acc, \
                [dia_clic_acc, dia_derm_acc, dia_fusion_acc], \
                sps_acc, \
                [sps_clic_acc, sps_derm_acc, sps_fusion_acc], \
                [pn_acc, str_acc, pig_acc, rs_acc, dag_acc, bwv_acc, vs_acc], \
                [[pn_clic_acc, pn_derm_acc, pn_fusion_acc], 
                 [str_clic_acc, str_derm_acc, str_fusion_acc], 
                 [pig_clic_acc, pig_derm_acc, pig_fusion_acc], 
                 [rs_clic_acc, rs_derm_acc, rs_fusion_acc], 
                 [dag_clic_acc, dag_derm_acc, dag_fusion_acc], 
                 [bwv_clic_acc, bwv_derm_acc, bwv_fusion_acc], 
                 [vs_clic_acc, vs_derm_acc, vs_fusion_acc]]