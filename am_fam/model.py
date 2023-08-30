import matplotlib.pyplot as plt
import torch.nn as nn
import torch, torchvision
from dalib.modules import WarmStartGradientReverseLayer
import warnings
warnings.filterwarnings('ignore')


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
    

# Attention-based reconstruction

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
        self.Sigmoid = nn.Sigmoid()#nn.Tanh()
        self.relu = nn.ReLU()
        self.final_conv =nn.Conv2d(32, 3, 3, padding=1) # , padding=1
        # self.final_conv1 =nn.Conv2d(32, 3, kernel_size = 1) # , padding=1
        self.seg_layers = nn.Sequential(self.up1, self.up2, self.up3)


    def forward(self, x):
        x = self.seg_layers(x)
        x = nn.functional.interpolate(x, size=self.output_size, mode='bicubic', align_corners=True)
        x = self.final_conv(x)
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