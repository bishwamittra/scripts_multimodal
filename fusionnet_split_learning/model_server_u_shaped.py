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


class FusionNet_server_middle(nn.Module):

    def __init__(self, architecture_choice):
        super(FusionNet_server_middle, self).__init__()
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
            self.dropout = nn.Dropout(0.3)

            # define the clinic model
            if(self.architecture_choice not in [2, 4]):
                self.layer1_cli = model_clinic.layer1
            self.layer2_cli = model_clinic.layer2
            self.layer3_cli = model_clinic.layer3
            self.layer4_cli = model_clinic.layer4
            self.avgpool_cli = model_clinic.avgpool
            # self.avgpool_cli = nn.MaxPool2d(7, 7)

            # define the derm model
            if(self.architecture_choice not in  [2, 4]):
                self.layer1_derm = model_derm.layer1        
            self.layer2_derm = model_derm.layer2
            self.layer3_derm = model_derm.layer3
            self.layer4_derm = model_derm.layer4
            self.avgpool_derm = model_derm.avgpool
            # self.avgpool_derm = nn.MaxPool2d(7, 7)
        
            if(self.architecture_choice not in [3, 4]):
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

        
    def forward(self, x):
        (x_clic, x_derm) = x

        if(self.architecture_choice in [1, 2, 3, 4]):


            # passed through the clinic model (pretained resnet50)
            if(self.architecture_choice not in [2, 4]):
                x_clic = self.layer1_cli(x_clic)
            x_clic = self.layer2_cli(x_clic)
            x_clic = self.layer3_cli(x_clic)
            x_clic = self.layer4_cli(x_clic)
            x_clic = self.avgpool_cli(x_clic)
            x_clic = x_clic.view(x_clic.size(0), -1)

            # passed through the derm model (pretained resnet50)
            if(self.architecture_choice not in  [2, 4]):
                x_derm = self.layer1_derm(x_derm)
            x_derm = self.layer2_derm(x_derm)
            x_derm = self.layer3_derm(x_derm)
            x_derm = self.layer4_derm(x_derm)
            x_derm = self.avgpool_derm(x_derm)
            x_derm = x_derm.view(x_derm.size(0), -1)

            # fusion of the two model outputs
            x_fusion = torch.add(x_clic, x_derm)
            if(self.architecture_choice not in [3, 4]):
                x_fusion = self.fc_fusion_(x_fusion)
                x_fusion = self.dropout(x_fusion)

                # mlp ahead of clinic model
                x_clic = self.clin_mlp(x_clic)
                x_clic = self.dropout(x_clic)

                # # mlp ahead of derm model
                x_derm = self.derm_mlp(x_derm)
                x_derm = self.dropout(x_derm)

        
        return (x_clic, x_derm, x_fusion)

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
