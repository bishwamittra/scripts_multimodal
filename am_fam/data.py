
import pandas as pd
import numpy as np
from PIL import Image
import os
import pandas as pd
from derm7pt.dataset import Derm7PtDataset, Derm7PtDatasetGroupInfrequent
# from derm7pt.vis import plot_confusion
# from derm7pt.kerasutils import deep_features
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')



dir_release = '../data/release_v0'
dir_meta = os.path.join(dir_release, 'meta')
dir_images = os.path.join(dir_release, 'images')
meta_df = pd.read_csv(os.path.join(dir_meta, 'meta.csv'))
train_indexes = list(pd.read_csv(os.path.join(dir_meta, 'train_indexes.csv'))['indexes'])
valid_indexes = list(pd.read_csv(os.path.join(dir_meta, 'valid_indexes.csv'))['indexes'])
test_indexes = list(pd.read_csv(os.path.join(dir_meta, 'test_indexes.csv'))['indexes'])
img_path = '../data/release_v0/images/'


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



class DermDataset(Dataset):
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