# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import pandas as pd
import numpy as np
from PIL import Image
pd.set_option('display.max_columns', 500)
import sys, os
from torch.utils.data import Dataset, DataLoader
import torch, torchvision
from torch.autograd import Function
from torch.autograd import Variable
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
from itertools import cycle
from itertools import chain
# from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from sklearn.metrics import classification_report
from imblearn.metrics import specificity_score
from dalib.modules import WarmStartGradientReverseLayer
import warnings
warnings.filterwarnings('ignore')

# %%
from derm7pt.dataset import Derm7PtDataset, Derm7PtDatasetGroupInfrequent
from derm7pt.vis import plot_confusion
from derm7pt.kerasutils import deep_features
dir_release = '../data/release_v0'
dir_meta = os.path.join(dir_release, 'meta')
dir_images = os.path.join(dir_release, 'images')
meta_df = pd.read_csv(os.path.join(dir_meta, 'meta.csv'))
train_indexes = list(pd.read_csv(os.path.join(dir_meta, 'train_indexes.csv'))['indexes'])
valid_indexes = list(pd.read_csv(os.path.join(dir_meta, 'valid_indexes.csv'))['indexes'])
test_indexes = list(pd.read_csv(os.path.join(dir_meta, 'test_indexes.csv'))['indexes'])

# %%
# The full dataset before any grouping of the labels.
derm_data = Derm7PtDataset(dir_images=dir_images, 
                        metadata_df=meta_df.copy(), # Copy as is modified.
                        train_indexes=train_indexes, valid_indexes=valid_indexes, 
                        test_indexes=test_indexes)

# The dataset after grouping infrequent labels.
derm_data_group = Derm7PtDatasetGroupInfrequent(dir_images=dir_images, 
                                             metadata_df=meta_df.copy(), # Copy as is modified.
                                             train_indexes=train_indexes, 
                                             valid_indexes=valid_indexes, 
                                             test_indexes=test_indexes)

# %%
derm_data.dataset_stats()

# %%
img_path = '../data/release_v0/images/'
BCC = ['basal cell carcinoma'] 
NEV = ['blue nevus', 'clark nevus', 'combined nevus', 'congenital nevus', 'dermal nevus', 'recurrent nevus', 'reed or spitz nevus']
MEL = ['melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)', 'melanoma (0.76 to 1.5 mm)', 'melanoma (more than 1.5 mm)', 'melanoma metastasis' ]
MISC = ['dermatofibroma', 'lentigo', 'melanosis', 'miscellaneous', 'vascular lesion']
SK = ['seborrheic keratosis']
PN = {'absent':0, 'typical':1, 'atypical':2}
STR = {'absent':0, 'regular':1, 'irregular':2}
PIG = {'absent':0, 'diffuse regular':1, 'localized regular':1, 'diffuse irregular':2, 'localized irregular':2}
RS = {'absent':0, 'blue areas':1, 'white areas':1, 'combinations':1}
DaG = {'absent':0, 'regular':1, 'irregular':2}
BWV = {'absent':0, 'present':1}
VS = {'absent':0, 'arborizing':1, 'comma':1, 'hairpin':1, 'within regression':1, 'wreath':1, 'dotted':2, 'linear irregular':2}
def get_diag_label(diag):
    if diag in BCC:
        label = 0
    elif diag in NEV:
        label = 1
    elif diag in MEL:
        label = 2
    elif diag in MISC:
        label = 3
    elif diag in SK:
        label = 4
    if label == None:
        print("Error!")
    else:
        return label
def get_7point_label(point_criteria):
    label0 = PN[point_criteria[0]]
    label1 = STR[point_criteria[1]]
    label2 = PIG[point_criteria[2]]
    label3 = RS[point_criteria[3]]
    label4 = DaG[point_criteria[4]]
    label5 = BWV[point_criteria[5]]
    label6 = VS[point_criteria[6]]
    return [label0, label1, label2, label3, label4, label5, label6]
clinic_train = []
clinic_validate = []
clinic_test = []
dermoscopic_train = []
dermoscopic_validate = []
dermoscopic_test = []
label_train_diag = []
label_validate_diag = []
label_test_diag = []
label_train_crit = []
label_validate_crit = []
label_test_crit = []
for index, row in meta_df.iterrows():
    c_img = row[15]
    d_img = row[16]
    diag = row[1]
    p_n = row[3]
    s_t_r = row[4]
    p_i_g = row[5]
    r_s = row[6]
    d_a_g = row[7]
    b_w_v = row[8]
    v_s = row[9]
    point_criteria = [p_n, s_t_r, p_i_g, r_s, d_a_g, b_w_v, v_s]
    # if d_img == 'FCl/Fcl068.jpg':
    #     d_img = 'FCL/Fcl068.jpg'

    if index in train_indexes:        
        clinic_train.append(img_path + c_img)
        dermoscopic_train.append(img_path + d_img)
        label_train_diag.append(get_diag_label(diag))
        label_train_crit.append(get_7point_label(point_criteria))
    elif index in valid_indexes:
        clinic_validate.append(img_path + c_img)
        dermoscopic_validate.append(img_path + d_img)
        label_validate_diag.append(get_diag_label(diag))
        label_validate_crit.append(get_7point_label(point_criteria))
    elif index in test_indexes:
        clinic_test.append(img_path + c_img)
        dermoscopic_test.append(img_path + d_img)
        label_test_diag.append(get_diag_label(diag))
        label_test_crit.append(get_7point_label(point_criteria))
    else:
        print("There is an error need to be fixed!")


# %%
# train_indexes.sort()
# valid_indexes.sort()
# print(train_indexes)
# print(valid_indexes)
# print(test_indexes)

# %%
print(len(clinic_train), len(clinic_validate), len(clinic_test))
# print(clinic_train)

# %%
class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        img_c = Image.open(self.data.c_path[idx])
        img_c = img_c.convert("RGB")
        img_c = self.transform(img_c)
        img_d = Image.open(self.data.d_path[idx])
        img_d = img_d.convert("RGB")
        img_d = self.transform(img_d)
        
        label_diag_i = np.array(self.data.lab_diag[idx])
        
        label_crit_i = np.array(self.data.lab_crit[idx])
        
        return img_c, img_d, label_diag_i, label_crit_i

# %%
# label_train_diag

# %%
class_sample_count = np.array([len(np.where(label_train_diag == t)[0]) for t in np.unique(label_train_diag)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in label_train_diag])

samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()
# sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# %%
train_transforms = transforms.Compose([transforms.Resize([299, 299]),
                                       transforms.Pad(padding=20, fill=(0, 0, 0)),
                                       transforms.RandomCrop([299, 299]),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomRotation([-45, 45]),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]) #(0.5 , 0.5 , 0.5), (0.5 , 0.5 , 0.5)
test_transforms = transforms.Compose([transforms.Resize([299, 299]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # (0.5 , 0.5 , 0.5), (0.5 , 0.5 , 0.5)
])
image_transforms = {'train':train_transforms, 'test':test_transforms}

train = list(zip(clinic_train, dermoscopic_train, label_train_diag, label_train_crit))
train_df = pd.DataFrame(train, columns=['c_path','d_path','lab_diag', 'lab_crit'])
train_dataset = MyDataset(train_df, transform=image_transforms['train'])

validate = list(zip(clinic_validate, dermoscopic_validate, label_validate_diag, label_validate_crit))
validate_df = pd.DataFrame(validate, columns=['c_path','d_path','lab_diag', 'lab_crit'])
validate_dataset = MyDataset(validate_df, transform=image_transforms['test'])

test = list(zip(clinic_test, dermoscopic_test, label_test_diag, label_test_crit))
test_df = pd.DataFrame(test, columns=['c_path','d_path','lab_diag', 'lab_crit'])
test_dataset = MyDataset(test_df, transform=image_transforms['test'])

# %%
def cal_auc(pre, true, show = False):
    auc_all = {}
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
        # Compute micro-average ROC curve and ROC area（方法二）
        # fpr["micro"], tpr["micro"], _ = roc_curve(np.array(one_hot).ravel(), np.array(pre[key]).ravel())
        fpr["micro"], tpr["micro"], _ = roc_curve(np.array(one_hot).ravel(), np.array(nn.Softmax(dim=1)(torch.Tensor(pre[key]))).ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area（方法一）
        # First aggregate all false positive rates
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
        
        if show == True:
            # Plot all ROC curves
            color_list = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'brown']
            lw=2
            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = cycle(color_list[0:n_classes])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
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

# %%
device = torch.device("cuda:0")# ("cuda:0")
class CNN(nn.Module): 
    def __init__(self, model):
        super(CNN, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        
    def forward(self, x):
        
        x = self.resnet_layer(x)
        return x
class Concate(nn.Module): 
    def __init__(self):
        super(Concate, self).__init__()
        self.hidden_size = 512 # 512
        
        self.avp_pooling = nn.AdaptiveAvgPool2d((1, 1)) # AdaptiveAvgPool2d
        self.linear_layer1 = nn.Linear(2048 * 2, self.hidden_size) # reduce the dimensional
        
        # attention computation using SEblock
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048, 2048 // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2048 // 2, 2048, bias=False),
            nn.Sigmoid()
        )
        self.att_relu = nn.ReLU()
        
        # define classifiers
        self.out_diag = nn.Linear(self.hidden_size, 5)
        self.out_crit_pn = nn.Linear(self.hidden_size, 3) # p_n 3, s_t_r 3, p_i_g 3, r_s 2, d_a_g 3, b_w_v 2, v_s 3
        self.out_crit_str = nn.Linear(self.hidden_size, 3)
        self.out_crit_pig = nn.Linear(self.hidden_size, 3)
        self.out_crit_rs = nn.Linear(self.hidden_size, 2)
        self.out_crit_dag = nn.Linear(self.hidden_size, 3)
        self.out_crit_bwv = nn.Linear(self.hidden_size, 2)
        self.out_crit_vs = nn.Linear(self.hidden_size, 3)
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, x_c, x_d):
        b, c, _, _ = x_c.size()
        img_size = torch.rand(x_c.size()[0], 3, x_c.size()[2], x_c.size()[3])
        # SE feature compute
        x_att_c = self.avg_pool(x_c).view(b, c)
        x_att_c = self.fc(x_att_c).view(b, c, 1, 1)
        x_att_feature_c = x_c * x_att_c.expand_as(x_c)
        
        x_att_d = self.avg_pool(x_d).view(b, c)
        x_att_d = self.fc(x_att_d).view(b, c, 1, 1)
        x_att_feature_d = x_d * x_att_d.expand_as(x_d)
        
        x_concat = torch.cat((x_c, x_d), dim=1)# x_att_feature_c, x_att_feature_d
         
        x_att_mask_c = torch.sum(x_c, dim=1) # x_att_feature_c
        x_att_mask_d = torch.sum(x_d, dim=1)
        # x_att_mask_c = x_att_c
        # x_att_mask_d = x_att_d
        x_att_mask_c = (x_att_mask_c - x_att_mask_c.min())/(x_att_mask_c.max() - x_att_mask_c.min()) #  (x - X_min) / (X_max - X_min)
        x_att_mask_d = (x_att_mask_d - x_att_mask_d.min())/(x_att_mask_d.max() - x_att_mask_d.min())
        # x_att_mask_c = torch.sum(x_att_feature_c, dim=1) / torch.sum(x_att_feature_c, dim=1).max()
        # x_att_mask_d = torch.sum(x_att_feature_d, dim=1) / torch.sum(x_att_feature_d, dim=1).max()
        x_att_mask_c = nn.functional.interpolate(x_att_mask_c.view(x_c.size()[0], 1, x_c.size()[2], x_c.size()[3]), size=(299, 299), mode='bilinear', align_corners=False) # bicubic, align_corners=False
        x_att_mask_d = nn.functional.interpolate(x_att_mask_d.view(x_d.size()[0], 1, x_d.size()[2], x_d.size()[3]), size=(299, 299), mode='bilinear', align_corners=False) # bicubic, align_corners=False
        
        # flatten feature vectors
        x = self.avp_pooling(x_concat).view(x_c.size()[0], -1)
        x = self.dropout(x)
        x = torch.relu(self.linear_layer1(x))
        x = self.dropout(x)
        # classifiers
        x_diag = self.out_diag(x)
        x_crit_pn = self.out_crit_pn(x)
        x_crit_str = self.out_crit_str(x)
        x_crit_pig = self.out_crit_pig(x)
        x_crit_rs = self.out_crit_rs(x)
        x_crit_dag = self.out_crit_dag(x)
        x_crit_bwv = self.out_crit_bwv(x)
        x_crit_vs = self.out_crit_vs(x)
        return x_diag, x_crit_pn, x_crit_str, x_crit_pig, x_crit_rs, x_crit_dag, x_crit_bwv, x_crit_vs, x_att_mask_c, x_att_mask_d, x_concat
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hidden_size = 256
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=100, auto_step=True) 
        self.hidden_layer = nn.Linear(2048, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
    def forward(self, x_c, x_d):
        x = torch.cat((x_c, x_d), dim = 0)
        x = self.avg_pool(x).view(x.size()[0], -1)
        x = self.grl(x)
        x = self.relu(self.hidden_layer(x))
        x = self.out(x)
        return x
    
resnet50 = models.resnet50(pretrained=True)
# resnet501 = models.resnet50(pretrained=True)
cnn_c = CNN(resnet50).to(device)
cnn_d = CNN(resnet50).to(device)
concate_net = Concate().to(device)
discriminator = Discriminator().to(device)# 判别分布

# %% [markdown]
# # Attention based reconstruction

# %%
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 0.0, 0.01)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
class ReconstructionNet(nn.Module):
    def __init__(self, in_feature, output_size):
        super(ReconstructionNet, self).__init__()
        self.output_size = output_size
        self.up1 = nn.Sequential(
                               nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True),
                               nn.Conv2d(in_feature, 128, 3, bias=False),
                               nn.BatchNorm2d(128),
                               nn.LeakyReLU(0.2),
        )
        self.up2 = nn.Sequential(
                               nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True),
                               nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                               nn.BatchNorm2d(64),
                               nn.LeakyReLU(0.2),

        )
        self.up3 = nn.Sequential(
                               nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
                               nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                               nn.BatchNorm2d(32),
                               nn.LeakyReLU(0.2),
        )
        self.Sigmoid = nn.Tanh() # nn.Sigmoid()
        self.relu = nn.ReLU()
        self.final_conv =nn.Conv2d(32, 3, 3, padding=1) # , padding=1
        # self.final_conv1 =nn.Conv2d(32, 3, kernel_size = 1) # , padding=1
        self.final_BN = nn.BatchNorm2d(3)
        self.seg_layers = nn.Sequential(self.up1, self.up2, self.up3)


    def forward(self, x):
        x = self.seg_layers(x)
        x = nn.functional.interpolate(x, size=self.output_size, mode='bicubic', align_corners=True)
        x = self.final_conv(x)
        x = self.final_BN(x)
        # x = self.final_conv1(x)
        y = self.Sigmoid(x)
        # y = (torch.sin(x) + 1)/2.
        return y

    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list

'''def reconstruction_loss(pred=None, ground_truth=None, mask=None, crit=None):
    pred1 = pred.view(pred.size()[0], -1)
    ground_truth1 = pred.view(ground_truth.size()[0], -1)
    
    loss = crit(pred.view(pred.size()[0], -1),  ground_truth.view(ground_truth.size()[0], -1))
    return loss '''
def reconstruction_loss(pred=None, ground_truth=None, mask=None, crit=None):
    '''import pdb;
    pdb.set_trace()'''
    pred1 = pred.view(pred.size()[0], -1)
    ground_truth1 = pred.view(ground_truth.size()[0], -1)
    
    loss = crit(pred.view(pred.size()[0], -1),  ground_truth.view(ground_truth.size()[0], -1))
    if mask != None:
        mask1 = torch.cat((mask, mask, mask), 1)
        mask1 = mask1.view(pred.size()[0], -1).detach()
        # weighted loss 
        loss = loss * torch.exp(mask1) # torch.exp()
    loss = torch.mean(torch.sum(loss, dim=1) / loss.size()[1]) # sum or mean
    return loss 
def show_reconstruction_batch(batch_img, mask = False): # show orignal images, attention maps and reconstruction images in one batch
    if mask == False:
        grid_img = torchvision.utils.make_grid(batch_img, nrow=4)
        plt.imshow(grid_img.permute(1, 2, 0).squeeze())
        plt.show()
    elif mask == True:
        # import pdb;pdb.set_trace() # torch.Size([13, 1, 299, 299])
        grid_img = torchvision.utils.make_grid(batch_img, nrow=4)
        plt.imshow(grid_img.permute(1, 2, 0).squeeze())
        plt.show()
    
reconstruct_net_c = ReconstructionNet(in_feature=2048 * 2, output_size=(299, 299)).to(device)
reconstruct_net_d = ReconstructionNet(in_feature=2048 * 2, output_size=(299, 299)).to(device)

# %%
import torch.optim.lr_scheduler as lr_scheduler
class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def get_scheduler(optimizer, lr_policy):
    """Return a learning rate scheduler
        Parameters:
        optimizer -- 网络优化器
        opt.lr_policy -- 学习率scheduler的名称: linear | step | plateau | cosine
    """
    # orch.optim.lr_scheduler.MultiStepLR
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        print("Using step schedular!")
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, threshold=1e-2, patience=4)
    elif lr_policy == 'cosine':
        print("Using cosine schedular!")
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-8)
    elif lr_policy == 'multi':
        print("Using multi step schedular!")
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 100], gamma=0.2) # (optimizer, milestones=[35, 80, 120], gamma=0.5)
    elif lr_policy == 'warmstart':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
    
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
def generate_label(batch_size):
    l_c = torch.zeros(batch_size)
    l_d = torch.ones(batch_size)
    label = torch.cat((l_c, l_d), dim=0)
    return label
def train_fun(dataloader, model_c, model_d, model_concate, model_recon_c, model_recon_d, epoch):
    model_c.train()
    model_d.train()
    model_concate.train()
    model_recon_c.train()
    model_recon_d.train()
    loss_all_count = []
    loss_diag_count = []
    loss_crit_pn_count = []
    loss_crit_str_count = []
    loss_crit_pig_count = []
    loss_crit_rs_count = []
    loss_crit_dag_count = []
    loss_crit_bwv_count = []
    loss_crit_vs_count = []
    loss_recon_c_count = []
    loss_recon_d_count = []
    loss_discriminator = []
    label_true = {'diag':[], 'pn':[], 'str':[], 'pig':[], 'rs':[], 'dag':[], 'bwv':[], 'vs':[]}
    pred_all = {'diag':[], 'pn':[], 'str':[], 'pig':[], 'rs':[], 'dag':[], 'bwv':[], 'vs':[]}
    for index, data in enumerate(dataloader):
        # optimizer_re.zero_grad()
        # optimizer_con.zero_grad()#
        img_c, img_d, lab_diag, lab_crit = data
        # print(lab_diag)
        label_true['diag'].extend(lab_diag)
        img_c, img_d, lab_diag = img_c.to(device), img_d.to(device), lab_diag.to(device)
        # import pdb;pdb.set_trace()
        label_true['pn'].extend(lab_crit[:, 0]), \
                    label_true['str'].extend(lab_crit[:, 1]), label_true['pig'].extend(lab_crit[:, 2]), label_true['rs'].extend(lab_crit[:, 3]), \
                    label_true['dag'].extend(lab_crit[:, 4]), label_true['bwv'].extend(lab_crit[:, 5]), label_true['vs'].extend(lab_crit[:, 6])
        lab_crit_pn, lab_crit_str, lab_crit_pig, lab_crit_rs, lab_crit_dag, lab_crit_bwv, lab_crit_vs = lab_crit[:, 0].to(device), lab_crit[:, 1].to(device), lab_crit[:, 2].to(device), lab_crit[:, 3].to(device), lab_crit[:, 4].to(device), lab_crit[:, 5].to(device), lab_crit[:, 6].to(device)
        
        #print(lab_crit_dag)
        feature_c = model_c(img_c)# predict for each class by using two modalities image through concatenate the features
        feature_d = model_d(img_d)# predict for each class by using two modalities image through concatenate the features
        prediction = model_concate(feature_c, feature_d)
        out_diag, out_crit_pn, out_crit_str, out_crit_pig, out_crit_rs, out_crit_dag, out_crit_bwv, out_crit_vs, \
                    att_mask_c, att_mask_d, att_feature = prediction
        
        # reconstrution
        recon_pred_c = model_recon_c(att_feature)
        recon_pred_d = model_recon_d(att_feature)
        recon_loss_c = reconstruction_loss(recon_pred_c, img_c, att_mask_c, crit=criterion_recon) # criterion_l1, criterion_recon
        recon_loss_d = reconstruction_loss(recon_pred_d, img_d, att_mask_d, crit=criterion_recon) # criterion_l1, criterion_recon
        
        # discriminator
        label_dis = generate_label(img_c.size()[0]).to(device, dtype=torch.int64)
        prediction_domain = discriminator(feature_c, feature_d)
        # import pdb;pdb.set_trace()
        loss_dis = criterion(prediction_domain, label_dis)
        # import pdb;pdb.set_trace()
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
        
        loss_all = 1/8 * (loss_diag + loss_crit_pn + loss_crit_str + loss_crit_pig + loss_crit_rs + loss_crit_dag + loss_crit_bwv \
                    + loss_crit_vs)  + 0.4 * (recon_loss_c + recon_loss_d) + 0.8 * loss_dis
        loss_all.backward()
        # optimizer_re.step()
        # optimizer_con.step()
        optimizer.step()
        loss_all_count.append(loss_all.item())
        loss_diag_count.append(loss_diag.item())
        loss_crit_pn_count.append(loss_crit_pn.item())
        loss_crit_str_count.append(loss_crit_str.item())
        loss_crit_pig_count.append(loss_crit_pig.item())
        loss_crit_rs_count.append(loss_crit_rs.item())
        loss_crit_dag_count.append(loss_crit_dag.item())
        loss_crit_bwv_count.append(loss_crit_bwv.item())
        loss_crit_vs_count.append(loss_crit_vs.item())
        loss_recon_c_count.append(recon_loss_c.item())
        loss_recon_d_count.append(recon_loss_d.item())
        loss_discriminator.append(loss_dis.item())
        
        optimizer.zero_grad()
    print("Epoch: {} train loss, Diag loss: {:.4f}, PN loss: {:.4f}, STR loss: {:.4f}, PIG loss: {:.4f}, RS loss: {:.4f}, DaG loss: {:.4f}, BWV loss: {:.4f}, VS loss: {:.4f}, discriminator loss: {:.4f}".format(
                    epoch, np.average(loss_diag_count), np.average(loss_crit_pn_count), np.average(loss_crit_str_count), np.average(loss_crit_pig_count), np.average(loss_crit_rs_count), np.average(loss_crit_dag_count), np.average(loss_crit_bwv_count), np.average(loss_crit_vs_count), np.average(loss_discriminator)))
    log_file.write("Epoch: {} train loss, Diag loss: {:.4f}, PN loss: {:.4f}, STR loss: {:.4f}, PIG loss: {:.4f}, RS loss: {:.4f}, DaG loss: {:.4f}, BWV loss: {:.4f}, VS loss: {:.4f}\n".format(
                    epoch, np.average(loss_diag_count), np.average(loss_crit_pn_count), np.average(loss_crit_str_count), np.average(loss_crit_pig_count), np.average(loss_crit_rs_count), np.average(loss_crit_dag_count), np.average(loss_crit_bwv_count), np.average(loss_crit_vs_count)))
    print("Reconstruction loss: c_loss: {:.4f}, d_loss: {:.4f}".format(np.average(loss_recon_c_count), np.average(loss_recon_d_count)))
    '''if epoch == 10:
        import pdb;pdb.set_trace()
        print("Debug!")'''
        
    if epoch % 5 == 0:
        # show the images 
        # clinical images
        show_reconstruction_batch((img_c.detach().cpu() + 1.)/2.)
        show_reconstruction_batch(att_mask_c.detach().cpu(), mask = True)
        show_reconstruction_batch(recon_pred_c.detach().cpu())
        # dermoscopic images
        show_reconstruction_batch((img_d.detach().cpu() + 1.)/2.)
        show_reconstruction_batch(att_mask_d.detach().cpu(), mask = True)
        show_reconstruction_batch(recon_pred_d.detach().cpu())
    return pred_all, label_true
def validate_fun(dataloader, model_c, model_d, model_concate, model_recon, model_recon_d, epoch):
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
        print("Epoch: {} validate loss, Diag loss: {:.4f}, PN loss: {:.4f}, STR loss: {:.4f}, PIG loss: {:.4f}, RS loss: {:.4f}, DaG loss: {:.4f}, BWV loss: {:.4f}, VS loss: {:.4f}".format(
                        epoch, np.average(loss_diag_count), np.average(loss_crit_pn_count), np.average(loss_crit_str_count), np.average(loss_crit_pig_count), np.average(loss_crit_rs_count), np.average(loss_crit_dag_count), np.average(loss_crit_bwv_count), np.average(loss_crit_vs_count)))
        log_file.write("Epoch: {} validate loss, Diag loss: {:.4f}, PN loss: {:.4f}, STR loss: {:.4f}, PIG loss: {:.4f}, RS loss: {:.4f}, DaG loss: {:.4f}, BWV loss: {:.4f}, VS loss: {:.4f}\n".format(
                        epoch, np.average(loss_diag_count), np.average(loss_crit_pn_count), np.average(loss_crit_str_count), np.average(loss_crit_pig_count), np.average(loss_crit_rs_count), np.average(loss_crit_dag_count), np.average(loss_crit_bwv_count), np.average(loss_crit_vs_count)))
    return pred_all, label_true

def test_fun(dataloader, model_c, model_d, model_concate, model_recon_c, model_recon_d, epoch):
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
        log_file.write("Epoch: {} test loss, Diag loss: {:.4f}, PN loss: {:.4f}, STR loss: {:.4f}, PIG loss: {:.4f}, RS loss: {:.4f}, DaG loss: {:.4f}, BWV loss: {:.4f}, VS loss: {:.4f}\n".format(
                        epoch, np.average(loss_diag_count), np.average(loss_crit_pn_count), np.average(loss_crit_str_count), np.average(loss_crit_pig_count), np.average(loss_crit_rs_count), np.average(loss_crit_dag_count), np.average(loss_crit_bwv_count), np.average(loss_crit_vs_count)))
        if epoch % 10 == 0:
            # show the images 
            # clinical images
            show_reconstruction_batch((img_c.detach().cpu() + 1.)/2.)
            show_reconstruction_batch(att_mask_c.detach().cpu(), mask = True)
            show_reconstruction_batch(recon_pred_c.detach().cpu())
            # dermoscopic images
            show_reconstruction_batch((img_d.detach().cpu() + 1.)/2.)
            show_reconstruction_batch(att_mask_d.detach().cpu(), mask = True)
            show_reconstruction_batch(recon_pred_d.detach().cpu())
    return pred_all, label_true

# %%
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

os.system("mkdir -p log")
log_file = open('./log/log' + 'concate_reconstruct_attention_fusion_new' + '.txt', 'w', buffering = 1)
epochs = 1
record_acc = 0.
record_auc = 0.

record_acc1 = 0.
record_auc1 = 0.


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
    print("\n\n\nEpoch {} begin training...".format(i))
    log_file.write("\n\nEpoch {} begin training...\n".format(i))
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
        # log_file.write(str(avg_acc))
        log_file.write("Current average ACC: {:.4f} \n".format(avg_acc))
        log_file.write("Current average AUC: {:.4f} \n".format(avg_auc))
        # log_file.write(str(avg_auc))

log_file.close()

# %%


# %%



