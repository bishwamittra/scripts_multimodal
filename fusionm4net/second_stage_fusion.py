# %%
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %%
import torch.nn.functional as F
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score, roc_curve,auc,average_precision_score,precision_recall_curve
import cv2
from dependency import *
import pandas as pd
from utils import encode_test_label,Logger,encode_meta_label
from tqdm import tqdm_notebook
#from plot import encode_meta_label    

test_index_df = pd.read_csv(test_index_path)
train_index_df = pd.read_csv(train_index_path)
val_index_df = pd.read_csv(val_index_path)

train_index_list = list(train_index_df['indexes'])
val_index_list = list(val_index_df['indexes'])
test_index_list = list(test_index_df['indexes'])

train_index_list_1 = train_index_list[0:206]
train_index_list_2 = train_index_list[206:]

df = pd.read_csv(img_info_path)

# %%
from keras.utils import to_categorical
from second_stage_fusion_utils import find_best_threshold, predict
import pandas as pd
from dependency import *
import torch
from model import FusionNet

# %%
def get_label_list(image_index_list):
    diag_label_list = []
    pn_label_list = []
    str_label_list = []
    pig_label_list = []
    rs_label_list = []
    dag_label_list = []
    bwv_label_list = []
    vs_label_list = []
    meta_list = []

    img_feature = []
    img_hf_feature = []
    img_vf_feature = []
    img_vhf_feature = []

    from sklearn.decomposition import PCA

    from tqdm import tqdm_notebook,tqdm
    for index_num in tqdm(image_index_list):
    #index_num = test_index_list[100]
        img_info = df[index_num:index_num+1]
        clinic_path = img_info['clinic']
        dermoscopy_path = img_info['derm']
        source_dir = '../release_v0/release_v0/images/'
        clinic_img = cv2.imread(source_dir+clinic_path[index_num])
        dermoscopy_img = cv2.imread(source_dir+dermoscopy_path[index_num])
        meta_vector,_,_ = encode_meta_label(img_info,index_num)

        [diagnosis_label,pigment_network_label,streaks_label,pigmentation_label,regression_structures_label,
         dots_and_globules_label,blue_whitish_veil_label, vascular_structures_label],[diagnosis_label_one_hot,pigment_network_label_one_hot,
        streaks_label_one_hot,pigmentation_label_one_hot,regression_structures_label_one_hot,
        dots_and_globules_label_one_hot,blue_whitish_veil_label_one_hot, vascular_structures_label_one_hot] = encode_test_label(img_info,index_num)

        diag_label_list.append(diagnosis_label)
        pn_label_list.append(pigment_network_label)
        str_label_list.append(streaks_label)
        pig_label_list.append(pigmentation_label)
        rs_label_list.append(regression_structures_label)
        dag_label_list.append(dots_and_globules_label)
        bwv_label_list.append(blue_whitish_veil_label)
        vs_label_list.append(vascular_structures_label)
        meta_list.append(meta_vector)



    label_dict ={'diag':diag_label_list,
                 'pn':pn_label_list,
                 'str':str_label_list,
                 'pig':pig_label_list,
                 'rs':rs_label_list,
                 'dag':dag_label_list,
                 'bwv':bwv_label_list,
                 'vs':vs_label_list}
    
    return label_dict,meta_list

# %%
def prediction_2_weight_search(vs_prob2,val_vs_preds_prob,val_img_vs_label):
    weight_list = []
    acc_list    = []
    for i in np.linspace(0,1,num=100):

        vs_prob_ = i*vs_prob2+(1-i)*val_vs_preds_prob
        val_vs_preds_fusion = np.argmax(vs_prob_,1)
        vs_acc = np.mean(val_vs_preds_fusion==val_img_vs_label)
       # print(vs_acc)
        weight_list.append(i)
        acc_list.append(vs_acc)
        
        
        index = np.argmax(acc_list)
        best_weight = weight_list[index]
        best_acc    = acc_list[index]
        
    return weight_list,acc_list,best_weight,best_acc

# %%
def save_gt_result(plot_dir,gt_list,prob_list):
    
    gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt,bwv_gt,vs_gt = gt_list
    prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob = prob_list    
    
    os.makedirs(plot_dir,exist_ok=True)
    
    np.save(plot_dir+'gt.npy',gt)
    np.save(plot_dir+'pn_gt.npy',pn_gt)
    np.save(plot_dir+'str_gt.npy',str_gt)
    np.save(plot_dir+'pig_gt.npy',pig_gt)
    np.save(plot_dir+'rs_gt.npy',rs_gt)
    np.save(plot_dir+'dag_gt.npy',dag_gt)
    np.save(plot_dir+'bwv_gt.npy',bwv_gt)
    np.save(plot_dir+'vs_gt.npy',vs_gt)
    
    print(vs_prob.shape)
    np.save(plot_dir+'prob.npy',prob)
    np.save(plot_dir+'pn_prob.npy',pn_prob)
    np.save(plot_dir+'str_prob.npy',str_prob)
    np.save(plot_dir+'pig_prob.npy',pig_prob)
    np.save(plot_dir+'rs_prob.npy',rs_prob)
    np.save(plot_dir+'dag_prob.npy',dag_prob)
    np.save(plot_dir+'bwv_prob.npy',bwv_prob)
    np.save(plot_dir+'vs_prob.npy',vs_prob)    

# %%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def classifier_cluster_predict(test_img_total_feature,classifier_name='SVM'):
    if classifier_name == 'SVM':
        #clf = SVC(C=1,kernel='rbf',probability=True)
        clf = SVC(C=1,kernel='rbf',probability=True)
        
    elif classifier_name == 'MLP':
        clf = MLPClassifier()
    elif classifier_name == 'Adaboost':
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                                 algorithm="SAMME",
                                 n_estimators=200, learning_rate=0.8) 
    elif classifier_name == 'LR':
        clf = LogisticRegression(C=0.01, class_weight='balanced')
    
    clf.fit(img_total_feature, img_diag_label)
    test_preds = clf.predict(test_img_total_feature)
    test_preds_prob = clf.predict_proba(test_img_total_feature)

    clf.fit(img_total_feature, img_pn_label)
    pn_test_preds = clf.predict(test_img_total_feature)
    test_pn_preds_prob = clf.predict_proba(test_img_total_feature)

    clf.fit(img_total_feature, img_str_label)
    str_test_preds = clf.predict(test_img_total_feature)
    test_str_preds_prob = clf.predict_proba(test_img_total_feature)

    clf.fit(img_total_feature, img_pig_label)
    pig_test_preds = clf.predict(test_img_total_feature)
    test_pig_preds_prob = clf.predict_proba(test_img_total_feature)

    clf.fit(img_total_feature, img_rs_label)
    rs_test_preds = clf.predict(test_img_total_feature)
    test_rs_preds_prob = clf.predict_proba(test_img_total_feature)

    clf.fit(img_total_feature, img_dag_label)
    dag_test_preds = clf.predict(test_img_total_feature)
    test_dag_preds_prob = clf.predict_proba(test_img_total_feature)

    clf.fit(img_total_feature, img_bwv_label)
    bwv_test_preds = clf.predict(test_img_total_feature)
    test_bwv_preds_prob = clf.predict_proba(test_img_total_feature)

    clf.fit(img_total_feature, img_vs_label)
    vs_test_preds = clf.predict(test_img_total_feature)
    test_vs_preds_prob = clf.predict_proba(test_img_total_feature)

    return [[test_preds,pn_test_preds,str_test_preds,pig_test_preds,rs_test_preds,dag_test_preds,bwv_test_preds,vs_test_preds],
            [test_preds_prob,test_pn_preds_prob,test_str_preds_prob,test_pig_preds_prob,
             test_rs_preds_prob,test_dag_preds_prob,test_bwv_preds_prob,test_vs_preds_prob]]

# %%
#hyperparameters
n_features = 20
search_model = ['FusionM4Net-FS']
mode = 'multimodal'
model_name  = 'FusionM4Net-FS'
search_num = 100 #after we optimize the speed of searching process, you can try to set search_num as 100. 
import random
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] =str(seed)
np.random.seed(seed)
TTA = 6

candidate_mode = 'Linear'
size = 224
round_ = 1
data_mode = 'Normal'
#for i in range(round_):
#i = 55
classifier_name = 'SVM'
p1_p2_fused_mode = 'searched'

p1_acc_list = []
p2_acc_list = []
p3_acc_list = []

mean_avg_acc_list = []

############################get test dataset predictions
print('predicting on testing dataset....P_1')
for j in range(round_):
    i =  100+j
    #weight_file = './{}_{}_{}_weight_file/{}/checkpoint/best_mean_acc_model.pth'.format(mode,model_name,data_mode,j)
    weight_file = './multimodal_FusionM4Net-FS_self_evaluated_weight_file/{}/checkpoint/best_mean_acc_model.pth'.format(j)

    print(weight_file)
    net = FusionNet(class_list).cuda()
    out_dir = './{}_{}_result/{}/'.format(mode,model_name, i)
    net.load_state_dict(torch.load(weight_file))

    plot_dir_p1 = './{}_{}_{}_p1_plot_figure/{}/'.format(mode, model_name,candidate_mode, i)

    if model_name in search_model:
        acc_array, weight_array, weight_list  = find_best_threshold(net,val_index_list,df,weight_file,
                                                                    model_name,out_dir,mode,search_num,TTA,size,candidate_mode)

    (avg_acc,
     [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob], 
     [nevu_label, bcc_label, mel_label, misc_label, sk_label],
     [nevu_prob ,bcc_prob ,  mel_prob ,misc_prob ,sk_prob],
     seven_point_feature_list,
     [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt] )= predict(net,test_index_list,df,weight_file,
                                                                    model_name,out_dir,mode,weight_list,TTA,size)

    print(i,weight_list)
    np.save('weight_list_{}_{}.npy'.format(candidate_mode,i),np.array(weight_list))
    p1_acc_list.append(avg_acc)
    prob_1 = prob
    pn_prob_1 = pn_prob
    str_prob_1 = str_prob
    pig_prob_1 = pig_prob
    rs_prob_1  = rs_prob
    dag_prob_1 = dag_prob
    bwv_prob_1 = bwv_prob
    vs_prob_1  = vs_prob
    seven_point_feature_list_p1 = seven_point_feature_list

    test_label_list = [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt]
    p1_list  = [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob]

    save_gt_result(plot_dir_p1,test_label_list,p1_list)
    print('Done!')

    #################get validation dataset predictions#####################
    print('predicting on validation dataset....P_val')


    (avg_acc,
     [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob], 
     [nevu_label, bcc_label, mel_label, misc_label, sk_label],
     [nevu_prob ,bcc_prob ,  mel_prob ,misc_prob ,sk_prob],
     seven_point_feature_list,
     [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt] )= predict(net,val_index_list,df,weight_file,
                                                                    model_name,out_dir,mode,weight_list,TTA,size)

    prob_val = prob
    pn_prob_val = pn_prob
    str_prob_val = str_prob
    pig_prob_val = pig_prob
    rs_prob_val  = rs_prob
    dag_prob_val = dag_prob
    bwv_prob_val = bwv_prob
    vs_prob_val  = vs_prob

    seven_point_feature_list_val = seven_point_feature_list
    print('Done!')


    #################get sub-test (divided from train) dataset predictions#####################
    print('predicting on sub-train (divided from train dataset) dataset....P_train')
    weight_file_evaluated = './{}_{}_{}_weight_file/{}/checkpoint/best_mean_acc_model.pth'.format(mode,model_name,'self_evaluated',
                                                                                       i)
    # weight_file_evaluated = './self_evaluated/{}/checkpoint/best_mean_acc_model.pth'.format(j)
    out_dir = './{}_{}_{}_result/{}/'.format(mode, model_name,candidate_mode, i)

    net.load_state_dict(torch.load(weight_file_evaluated))

    (avg_acc,
     [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob], 
     [nevu_label, bcc_label, mel_label, misc_label, sk_label],
     [nevu_prob ,bcc_prob ,  mel_prob ,misc_prob ,sk_prob],
     seven_point_feature_list,
     [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt] )= predict(net,train_index_list_2,df,weight_file,
                                                                    model_name,out_dir,mode,weight_list,TTA,size)



    prob_self_evaluated = prob
    pn_prob_self_evaluated = pn_prob
    str_prob_self_evaluated = str_prob
    pig_prob_self_evaluated = pig_prob
    rs_prob_self_evaluated  = rs_prob
    dag_prob_self_evaluated = dag_prob
    bwv_prob_self_evaluated = bwv_prob
    vs_prob_self_evaluated  = vs_prob

    seven_point_feature_list_self_evaluated = seven_point_feature_list


    #################get labels#####################
    print('fusing fusing predictions from two-modality images and metadata......')
    train_label_dict,train_meta_list = get_label_list(train_index_list_2)
    val_label_dict,val_meta_list   = get_label_list(val_index_list)
    test_label_dict,test_meta_list  = get_label_list(test_index_list)



    #################fusing predictions from two-modality images and metadata by concatenation#####################
    train_meta_array = np.array(train_meta_list)
    img_total_feature = np.concatenate([
                                  seven_point_feature_list_self_evaluated,
                                  train_meta_array      
                                  ],1)
    print(img_total_feature.shape)
    train_diag_label = np.array(train_label_dict['diag'])
    print(train_diag_label.shape)

    val_meta_array = np.array(val_meta_list)
    val_img_total_feature = np.concatenate([
                                 seven_point_feature_list_val,
                                 val_meta_array      
                                  ],1)

    print(val_img_total_feature.shape)
    val_diag_label = np.array(val_label_dict['diag'])
    print(val_diag_label.shape)

    test_meta_array = np.array(test_meta_list)
    test_img_total_feature = np.concatenate([
                                  seven_point_feature_list_p1,
                                  test_meta_array      
                                  ],1)
    
    print(test_img_total_feature.shape)
    test_diag_label = np.array(test_label_dict['diag'])
    print(test_diag_label.shape)
    print('Done!')

    img_pn_label= np.array(train_label_dict['pn'])
    img_str_label= np.array(train_label_dict['str'])
    img_pig_label= np.array(train_label_dict['pig'])
    img_rs_label= np.array(train_label_dict['rs'])
    img_dag_label= np.array(train_label_dict['dag'])
    img_bwv_label= np.array(train_label_dict['bwv'])
    img_vs_label= np.array(train_label_dict['vs'])
    img_diag_label = np.array(train_label_dict['diag'])

    test_img_pn_label= np.array(test_label_dict['pn'])
    test_img_str_label= np.array(test_label_dict['str'])
    test_img_pig_label= np.array(test_label_dict['pig'])
    test_img_rs_label= np.array(test_label_dict['rs'])
    test_img_dag_label= np.array(test_label_dict['dag'])
    test_img_bwv_label= np.array(test_label_dict['bwv'])
    test_img_vs_label= np.array(test_label_dict['vs'])
    test_img_diag_label = np.array(test_label_dict['diag'])


    val_img_pn_label= np.array(val_label_dict['pn'])
    val_img_str_label= np.array(val_label_dict['str'])
    val_img_pig_label= np.array(val_label_dict['pig'])
    val_img_rs_label= np.array(val_label_dict['rs'])
    val_img_dag_label= np.array(val_label_dict['dag'])
    val_img_bwv_label= np.array(val_label_dict['bwv'])
    val_img_vs_label= np.array(val_label_dict['vs'])
    val_img_diag_label= np.array(val_label_dict['diag'])


    img_pn_label_one_hot = to_categorical(np.array(train_label_dict['pn']))
    img_str_label_one_hot= to_categorical(np.array(train_label_dict['str']))
    img_pig_labe_one_hotl= to_categorical(np.array(train_label_dict['pig']))
    img_rs_label_one_hot= to_categorical(np.array(train_label_dict['rs']))
    img_dag_label_one_hot= to_categorical(np.array(train_label_dict['dag']))
    img_bwv_label_one_hot= to_categorical(np.array(train_label_dict['bwv']))
    img_vs_label_one_hot= to_categorical(np.array(train_label_dict['vs']))

    test_img_pn_label_one_hot = to_categorical(np.array(test_label_dict['pn']))
    test_img_str_label_one_hot= to_categorical(np.array(test_label_dict['str']))
    test_img_pig_labe_one_hotl= to_categorical(np.array(test_label_dict['pig']))
    test_img_rs_label_one_hot= to_categorical(np.array(test_label_dict['rs']))
    test_img_dag_label_one_hot= to_categorical(np.array(test_label_dict['dag']))
    test_img_bwv_label_one_hot= to_categorical(np.array(test_label_dict['bwv']))
    test_img_vs_label_one_hot= to_categorical(np.array(test_label_dict['vs']))

    print('Training Classifier Cluster..... Classifier：{}'.format(classifier_name))
    _,[prob_2,pn_prob_2,str_prob_2,pig_prob_2,rs_prob_2,dag_prob_2,bwv_prob_2,vs_prob_2] = classifier_cluster_predict(test_img_total_feature,classifier_name)
    _,[val_p2_prob,val_p2_pn_prob,val_p2_str_prob,val_p2_pig_prob,val_p2_rs_prob,val_p2_dag_prob,val_p2_bwv_prob,val_p2_vs_prob] = classifier_cluster_predict(val_img_total_feature,classifier_name)
    print('Done!')
    p2_list  = [prob_2,pn_prob_2,str_prob_2,pig_prob_2,rs_prob_2,dag_prob_2,bwv_prob_2,vs_prob_2]
    plot_dir_p2 = './{}_{}_{}_p2_plot_figure/{}/'.format( mode,model_name, 
                                                                  candidate_mode, i)
    save_gt_result(plot_dir_p2,test_label_list,p2_list)


    pn_test_preds = np.argmax(pn_prob_2,1)
    str_test_preds = np.argmax(str_prob_2,1)
    pig_test_preds = np.argmax(pig_prob_2,1)
    rs_test_preds = np.argmax(rs_prob_2,1)
    dag_test_preds = np.argmax(dag_prob_2,1)
    bwv_test_preds = np.argmax(bwv_prob_2,1)
    vs_test_preds = np.argmax(vs_prob_2,1)
    diag_test_preds = np.argmax(prob_2,1)

    pn_acc = np.mean(pn_test_preds==test_img_pn_label)
    str_acc = np.mean(str_test_preds==test_img_str_label)
    pig_acc = np.mean(pig_test_preds==test_img_pig_label)
    rs_acc = np.mean(rs_test_preds==test_img_rs_label)
    dag_acc = np.mean(dag_test_preds==test_img_dag_label)
    bwv_acc = np.mean(bwv_test_preds==test_img_bwv_label)
    vs_acc = np.mean(vs_test_preds==test_img_vs_label)
    diag_acc = np.mean(diag_test_preds==test_img_diag_label)
    avg_acc = (pn_acc + str_acc + pig_acc + rs_acc + dag_acc + bwv_acc + vs_acc + diag_acc) / 8

    print('P2--------------------------')
    print('avg_acc : {}'.format(avg_acc))
    print('vs_acc : {}'.format(vs_acc))
    print('bwv_acc : {}'.format(bwv_acc))
    print('dag_acc : {}'.format(dag_acc))
    print('rs_acc : {}'.format(rs_acc))
    print('pig_acc : {}'.format(pig_acc))
    print('str_acc : {}'.format(str_acc))
    print('pn_acc : {}'.format(pn_acc))
    print('diag_acc : {}'.format(diag_acc))



    vs_weight_list,vs_acc_list,vs_best_weight,vs_best_acc = prediction_2_weight_search(vs_prob_val,val_p2_vs_prob,val_img_vs_label)
    str_weight_list,str_acc_list,str_best_weight,str_best_acc = prediction_2_weight_search(str_prob_val,val_p2_str_prob,val_img_str_label)
    pn_weight_list,pn_acc_list,pn_best_weight,pn_best_acc = prediction_2_weight_search(pn_prob_val,val_p2_pn_prob,val_img_pn_label)
    pig_weight_list,pig_acc_list,pig_best_weight,pig_best_acc = prediction_2_weight_search(pig_prob_val,val_p2_pig_prob,val_img_pig_label)
    bwv_weight_list,bwv_acc_list,bwv_best_weight,bwv_best_acc = prediction_2_weight_search(bwv_prob_val,val_p2_bwv_prob,val_img_bwv_label)
    dag_weight_list,dag_acc_list,dag_best_weight,dag_best_acc = prediction_2_weight_search(dag_prob_val,val_p2_dag_prob,val_img_dag_label)
    rs_weight_list,rs_acc_list,rs_best_weight,rs_best_acc = prediction_2_weight_search(rs_prob_val,val_p2_rs_prob,val_img_rs_label)
    pred_weight_list,pred_acc_list,pred_best_weight,pred_best_acc = prediction_2_weight_search(prob_val,val_p2_prob,val_img_diag_label)

    vs_arr = np.array(vs_acc_list)
    str_arr = np.array(str_acc_list)
    pn_arr = np.array(pn_acc_list)
    pig_arr = np.array(pig_acc_list)
    bwv_arr = np.array(bwv_acc_list)
    dag_arr = np.array(dag_acc_list)
    rs_arr = np.array(rs_acc_list)
    pred_arr = np.array(pred_acc_list)

    avg_arr = (vs_arr+str_arr+pn_arr+pig_arr+bwv_arr+dag_arr+rs_arr+pred_arr)/8
    index = np.argmax(avg_arr)

    print('searched P1-P2 fused weights----')
    print(pred_weight_list[index])
    print(avg_arr[index])

    best_weight = pred_weight_list[index]
    p1_p2_weight_list = [0.5,best_weight]
    p1_p2_fused_mode_list = ['average','searched']

    for index in range(len(p1_p2_weight_list)):

        best_weight = p1_p2_weight_list[index]
        p1_p2_fused_mode = p1_p2_fused_mode_list[index]

        prob_3 = (best_weight*prob_1+(1-best_weight)*prob_2)
        pred_3 = np.argmax(prob_3,1)

        pn_prob_3 = (best_weight*pn_prob_1 + (1-best_weight)*pn_prob_2)
        pn_pred_3 = np.argmax(pn_prob_3,1)

        pig_prob_3 = (best_weight*pig_prob_1 +(1-best_weight)*pig_prob_2)
        pig_pred_3 = np.argmax(pig_prob_3,1)

        str_prob_3 = (best_weight*str_prob_1 + (1-best_weight)*str_prob_2)
        str_pred_3 = np.argmax(str_prob_3,1)

        rs_prob_3 = (best_weight*rs_prob_1 + (1-best_weight)*rs_prob_2)
        rs_pred_3 = np.argmax(rs_prob_3,1)

        bwv_prob_3 = (best_weight*bwv_prob_1 + (1-best_weight)*bwv_prob_2)
        bwv_pred_3 = np.argmax(bwv_prob_3,1)

        vs_prob_3 = (best_weight*vs_prob_1 + (1-best_weight)*vs_prob_2)
        vs_pred_3 = np.argmax(vs_prob_3,1)

        dag_prob_3 = (best_weight*dag_prob_1 + (1-best_weight)*dag_prob_2)
        dag_pred_3 = np.argmax(dag_prob_3,1)

        pn_acc = np.mean(pn_pred_3==test_img_pn_label)
        str_acc = np.mean(str_pred_3==test_img_str_label)
        pig_acc = np.mean(pig_pred_3==test_img_pig_label)
        rs_acc = np.mean(rs_pred_3==test_img_rs_label)
        dag_acc = np.mean(dag_pred_3==test_img_dag_label)
        bwv_acc = np.mean(bwv_pred_3==test_img_bwv_label)
        vs_acc = np.mean(vs_pred_3==test_img_vs_label)
        diag_acc = np.mean(pred_3==test_img_diag_label)

        avg_acc = (pn_acc + str_acc + pig_acc + rs_acc + dag_acc + bwv_acc + vs_acc + diag_acc) / 8
        mean_avg_acc_list.append(avg_acc)
        p3_list  = [prob_3,pn_prob_3,str_prob_3,pig_prob_3,rs_prob_3,dag_prob_3,bwv_prob_3,vs_prob_3]
        plot_dir_p3 = './{}_{}_{}_{}_{}_p3_plot_figure/{}/'.format(model_name, mode,
                                                                      classifier_name, p1_p2_fused_mode, candidate_mode,
                                                                         i)
        save_gt_result(plot_dir_p3,test_label_list,p3_list)
        print('p1_p2_fused_mode : {}'.format(p1_p2_fused_mode))
        print('P3--------------------------')
        print('avg_acc : {}'.format(avg_acc))
        print('vs_acc : {}'.format(vs_acc))
        print('bwv_acc : {}'.format(bwv_acc))
        print('dag_acc : {}'.format(dag_acc))
        print('rs_acc : {}'.format(rs_acc))
        print('pig_acc : {}'.format(pig_acc))
        print('str_acc : {}'.format(str_acc))
        print('pn_acc : {}'.format(pn_acc))
        print('diag_acc : {}'.format(diag_acc))

averaged_avg_acc = np.mean([mean_avg_acc_list[0],mean_avg_acc_list[2],mean_avg_acc_list[4],mean_avg_acc_list[6],mean_avg_acc_list[8]])
searched_avg_acc = np.mean([mean_avg_acc_list[1],mean_avg_acc_list[3],mean_avg_acc_list[5],mean_avg_acc_list[7],mean_avg_acc_list[9]])

print('averaged avg acc: {}'.format(averaged_avg_acc))
print('searched avg acc: {}'.format(searched_avg_acc))


# %%
#hyperparameters
n_features = 20
search_model = ['FusionM4Net-FS']
mode = 'multimodal'
model_name  = 'FusionM4Net-FS'
search_num = 50
TTA = 6

candidate_mode = 'Linear'
size = 224
round_ = 5
data_mode = 'Normal'
#for i in range(round_):
#i = 55
classifier_name = 'SVM'
p1_p2_fused_mode = 'searched'

p1_acc_list = []
p2_acc_list = []
p3_acc_list = []

mean_avg_acc_list = []

############################get test dataset predictions
print('predicting on testing dataset....P_1')
for j in range(round_):
    i =  100+j
    #weight_file = './{}_{}_{}_weight_file/{}/checkpoint/best_mean_acc_model.pth'.format(mode,model_name,data_mode,j)
    weight_file = './weight/{}/checkpoint/best_mean_acc_model.pth'.format(j)

    print(weight_file)
    net = FusionNet(class_list).cuda()
    out_dir = './{}_{}_result/{}/'.format(mode,model_name, i)
    net.load_state_dict(torch.load(weight_file))

    plot_dir_p1 = './{}_{}_{}_p1_plot_figure/{}/'.format(mode, model_name,candidate_mode, i)

    if model_name in search_model:
        acc_array, weight_array, weight_list  = find_best_threshold(net,val_index_list,df,weight_file,
                                                                    model_name,out_dir,mode,search_num,TTA,size,candidate_mode)

    (avg_acc,
     [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob], 
     [nevu_label, bcc_label, mel_label, misc_label, sk_label],
     [nevu_prob ,bcc_prob ,  mel_prob ,misc_prob ,sk_prob],
     seven_point_feature_list,
     [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt] )= predict(net,test_index_list,df,weight_file,
                                                                    model_name,out_dir,mode,weight_list,TTA,size)

    print(i,weight_list)
    np.save('weight_list_{}_{}.npy'.format(candidate_mode,i),np.array(weight_list))
    p1_acc_list.append(avg_acc)
    prob_1 = prob
    pn_prob_1 = pn_prob
    str_prob_1 = str_prob
    pig_prob_1 = pig_prob
    rs_prob_1  = rs_prob
    dag_prob_1 = dag_prob
    bwv_prob_1 = bwv_prob
    vs_prob_1  = vs_prob
    seven_point_feature_list_p1 = seven_point_feature_list

    test_label_list = [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt]
    p1_list  = [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob]

    save_gt_result(plot_dir_p1,test_label_list,p1_list)
    print('Done!')

    #################get validation dataset predictions#####################
    print('predicting on validation dataset....P_val')


    (avg_acc,
     [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob], 
     [nevu_label, bcc_label, mel_label, misc_label, sk_label],
     [nevu_prob ,bcc_prob ,  mel_prob ,misc_prob ,sk_prob],
     seven_point_feature_list,
     [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt] )= predict(net,val_index_list,df,weight_file,
                                                                    model_name,out_dir,mode,weight_list,TTA,size)

    prob_val = prob
    pn_prob_val = pn_prob
    str_prob_val = str_prob
    pig_prob_val = pig_prob
    rs_prob_val  = rs_prob
    dag_prob_val = dag_prob
    bwv_prob_val = bwv_prob
    vs_prob_val  = vs_prob

    seven_point_feature_list_val = seven_point_feature_list
    print('Done!')

    #################get sub-test (divided from train) dataset predictions#####################
    print('predicting on sub-train (divided from train dataset) dataset....P_train')
    #weight_file_evaluated = './{}_{}_{}_weight_file/{}/checkpoint/best_mean_acc_model.pth'.format(mode,model_name,'self_evaluated',
    #                                                                                   i)
    weight_file_evaluated = './self_evaluated/{}/checkpoint/best_mean_acc_model.pth'.format(j)
    out_dir = './{}_{}_{}_result/{}/'.format(mode, model_name,candidate_mode, i)

    net.load_state_dict(torch.load(weight_file_evaluated))

    (avg_acc,
     [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob], 
     [nevu_label, bcc_label, mel_label, misc_label, sk_label],
     [nevu_prob ,bcc_prob ,  mel_prob ,misc_prob ,sk_prob],
     seven_point_feature_list,
     [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt] )= predict(net,train_index_list_2,df,weight_file,
                                                                    model_name,out_dir,mode,weight_list,TTA,size)



    prob_self_evaluated = prob
    pn_prob_self_evaluated = pn_prob
    str_prob_self_evaluated = str_prob
    pig_prob_self_evaluated = pig_prob
    rs_prob_self_evaluated  = rs_prob
    dag_prob_self_evaluated = dag_prob
    bwv_prob_self_evaluated = bwv_prob
    vs_prob_self_evaluated  = vs_prob

    seven_point_feature_list_self_evaluated = seven_point_feature_list


    #################get labels#####################
    print('fusing fusing predictions from two-modality images and metadata......')
    train_label_dict,train_meta_list = get_label_list(train_index_list_2)
    val_label_dict,val_meta_list   = get_label_list(val_index_list)
    test_label_dict,test_meta_list  = get_label_list(test_index_list)



    #################fusing predictions from two-modality images and metadata by concatenation#####################
    train_meta_array = np.array(train_meta_list)
    img_total_feature = np.concatenate([
                                  seven_point_feature_list_self_evaluated,
                                  train_meta_array      
                                  ],1)
    print(img_total_feature.shape)
    train_diag_label = np.array(train_label_dict['diag'])
    print(train_diag_label.shape)

    val_meta_array = np.array(val_meta_list)
    val_img_total_feature = np.concatenate([
                                 seven_point_feature_list_val,
                                 val_meta_array      
                                  ],1)

    print(val_img_total_feature.shape)
    val_diag_label = np.array(val_label_dict['diag'])
    print(val_diag_label.shape)

    test_meta_array = np.array(test_meta_list)
    test_img_total_feature = np.concatenate([
                                  seven_point_feature_list_p1,
                                  test_meta_array      
                                  ],1)
    
    print(test_img_total_feature.shape)
    test_diag_label = np.array(test_label_dict['diag'])
    print(test_diag_label.shape)
    print('Done!')

    img_pn_label= np.array(train_label_dict['pn'])
    img_str_label= np.array(train_label_dict['str'])
    img_pig_label= np.array(train_label_dict['pig'])
    img_rs_label= np.array(train_label_dict['rs'])
    img_dag_label= np.array(train_label_dict['dag'])
    img_bwv_label= np.array(train_label_dict['bwv'])
    img_vs_label= np.array(train_label_dict['vs'])
    img_diag_label = np.array(train_label_dict['diag'])

    test_img_pn_label= np.array(test_label_dict['pn'])
    test_img_str_label= np.array(test_label_dict['str'])
    test_img_pig_label= np.array(test_label_dict['pig'])
    test_img_rs_label= np.array(test_label_dict['rs'])
    test_img_dag_label= np.array(test_label_dict['dag'])
    test_img_bwv_label= np.array(test_label_dict['bwv'])
    test_img_vs_label= np.array(test_label_dict['vs'])
    test_img_diag_label = np.array(test_label_dict['diag'])


    val_img_pn_label= np.array(val_label_dict['pn'])
    val_img_str_label= np.array(val_label_dict['str'])
    val_img_pig_label= np.array(val_label_dict['pig'])
    val_img_rs_label= np.array(val_label_dict['rs'])
    val_img_dag_label= np.array(val_label_dict['dag'])
    val_img_bwv_label= np.array(val_label_dict['bwv'])
    val_img_vs_label= np.array(val_label_dict['vs'])
    val_img_diag_label= np.array(val_label_dict['diag'])


    img_pn_label_one_hot = to_categorical(np.array(train_label_dict['pn']))
    img_str_label_one_hot= to_categorical(np.array(train_label_dict['str']))
    img_pig_labe_one_hotl= to_categorical(np.array(train_label_dict['pig']))
    img_rs_label_one_hot= to_categorical(np.array(train_label_dict['rs']))
    img_dag_label_one_hot= to_categorical(np.array(train_label_dict['dag']))
    img_bwv_label_one_hot= to_categorical(np.array(train_label_dict['bwv']))
    img_vs_label_one_hot= to_categorical(np.array(train_label_dict['vs']))

    test_img_pn_label_one_hot = to_categorical(np.array(test_label_dict['pn']))
    test_img_str_label_one_hot= to_categorical(np.array(test_label_dict['str']))
    test_img_pig_labe_one_hotl= to_categorical(np.array(test_label_dict['pig']))
    test_img_rs_label_one_hot= to_categorical(np.array(test_label_dict['rs']))
    test_img_dag_label_one_hot= to_categorical(np.array(test_label_dict['dag']))
    test_img_bwv_label_one_hot= to_categorical(np.array(test_label_dict['bwv']))
    test_img_vs_label_one_hot= to_categorical(np.array(test_label_dict['vs']))

    print('Training Classifier Cluster..... Classifier：{}'.format(classifier_name))
    _,[prob_2,pn_prob_2,str_prob_2,pig_prob_2,rs_prob_2,dag_prob_2,bwv_prob_2,vs_prob_2] = classifier_cluster_predict(test_img_total_feature,classifier_name)
    _,[val_p2_prob,val_p2_pn_prob,val_p2_str_prob,val_p2_pig_prob,val_p2_rs_prob,val_p2_dag_prob,val_p2_bwv_prob,val_p2_vs_prob] = classifier_cluster_predict(val_img_total_feature,classifier_name)
    print('Done!')
    p2_list  = [prob_2,pn_prob_2,str_prob_2,pig_prob_2,rs_prob_2,dag_prob_2,bwv_prob_2,vs_prob_2]
    plot_dir_p2 = './{}_{}_{}_p2_plot_figure/{}/'.format( mode,model_name, 
                                                                  candidate_mode, i)
    save_gt_result(plot_dir_p2,test_label_list,p2_list)


    pn_test_preds = np.argmax(pn_prob_2,1)
    str_test_preds = np.argmax(str_prob_2,1)
    pig_test_preds = np.argmax(pig_prob_2,1)
    rs_test_preds = np.argmax(rs_prob_2,1)
    dag_test_preds = np.argmax(dag_prob_2,1)
    bwv_test_preds = np.argmax(bwv_prob_2,1)
    vs_test_preds = np.argmax(vs_prob_2,1)
    diag_test_preds = np.argmax(prob_2,1)

    pn_acc = np.mean(pn_test_preds==test_img_pn_label)
    str_acc = np.mean(str_test_preds==test_img_str_label)
    pig_acc = np.mean(pig_test_preds==test_img_pig_label)
    rs_acc = np.mean(rs_test_preds==test_img_rs_label)
    dag_acc = np.mean(dag_test_preds==test_img_dag_label)
    bwv_acc = np.mean(bwv_test_preds==test_img_bwv_label)
    vs_acc = np.mean(vs_test_preds==test_img_vs_label)
    diag_acc = np.mean(diag_test_preds==test_img_diag_label)
    avg_acc = (pn_acc + str_acc + pig_acc + rs_acc + dag_acc + bwv_acc + vs_acc + diag_acc) / 8

    print('P2--------------------------')
    print('avg_acc : {}'.format(avg_acc))
    print('vs_acc : {}'.format(vs_acc))
    print('bwv_acc : {}'.format(bwv_acc))
    print('dag_acc : {}'.format(dag_acc))
    print('rs_acc : {}'.format(rs_acc))
    print('pig_acc : {}'.format(pig_acc))
    print('str_acc : {}'.format(str_acc))
    print('pn_acc : {}'.format(pn_acc))
    print('diag_acc : {}'.format(diag_acc))



    vs_weight_list,vs_acc_list,vs_best_weight,vs_best_acc = prediction_2_weight_search(vs_prob_val,val_p2_vs_prob,val_img_vs_label)
    str_weight_list,str_acc_list,str_best_weight,str_best_acc = prediction_2_weight_search(str_prob_val,val_p2_str_prob,val_img_str_label)
    pn_weight_list,pn_acc_list,pn_best_weight,pn_best_acc = prediction_2_weight_search(pn_prob_val,val_p2_pn_prob,val_img_pn_label)
    pig_weight_list,pig_acc_list,pig_best_weight,pig_best_acc = prediction_2_weight_search(pig_prob_val,val_p2_pig_prob,val_img_pig_label)
    bwv_weight_list,bwv_acc_list,bwv_best_weight,bwv_best_acc = prediction_2_weight_search(bwv_prob_val,val_p2_bwv_prob,val_img_bwv_label)
    dag_weight_list,dag_acc_list,dag_best_weight,dag_best_acc = prediction_2_weight_search(dag_prob_val,val_p2_dag_prob,val_img_dag_label)
    rs_weight_list,rs_acc_list,rs_best_weight,rs_best_acc = prediction_2_weight_search(rs_prob_val,val_p2_rs_prob,val_img_rs_label)
    pred_weight_list,pred_acc_list,pred_best_weight,pred_best_acc = prediction_2_weight_search(prob_val,val_p2_prob,val_img_diag_label)

    vs_arr = np.array(vs_acc_list)
    str_arr = np.array(str_acc_list)
    pn_arr = np.array(pn_acc_list)
    pig_arr = np.array(pig_acc_list)
    bwv_arr = np.array(bwv_acc_list)
    dag_arr = np.array(dag_acc_list)
    rs_arr = np.array(rs_acc_list)
    pred_arr = np.array(pred_acc_list)

    avg_arr = (vs_arr+str_arr+pn_arr+pig_arr+bwv_arr+dag_arr+rs_arr+pred_arr)/8
    index = np.argmax(avg_arr)

    print('searched P1-P2 fused weights----')
    print(pred_weight_list[index])
    print(avg_arr[index])

    best_weight = pred_weight_list[index]
    p1_p2_weight_list = [0.5,best_weight]
    p1_p2_fused_mode_list = ['average','searched']

    for index in range(len(p1_p2_weight_list)):

        best_weight = p1_p2_weight_list[index]
        p1_p2_fused_mode = p1_p2_fused_mode_list[index]

        prob_3 = (best_weight*prob_1+(1-best_weight)*prob_2)
        pred_3 = np.argmax(prob_3,1)

        pn_prob_3 = (best_weight*pn_prob_1 + (1-best_weight)*pn_prob_2)
        pn_pred_3 = np.argmax(pn_prob_3,1)

        pig_prob_3 = (best_weight*pig_prob_1 +(1-best_weight)*pig_prob_2)
        pig_pred_3 = np.argmax(pig_prob_3,1)

        str_prob_3 = (best_weight*str_prob_1 + (1-best_weight)*str_prob_2)
        str_pred_3 = np.argmax(str_prob_3,1)

        rs_prob_3 = (best_weight*rs_prob_1 + (1-best_weight)*rs_prob_2)
        rs_pred_3 = np.argmax(rs_prob_3,1)

        bwv_prob_3 = (best_weight*bwv_prob_1 + (1-best_weight)*bwv_prob_2)
        bwv_pred_3 = np.argmax(bwv_prob_3,1)

        vs_prob_3 = (best_weight*vs_prob_1 + (1-best_weight)*vs_prob_2)
        vs_pred_3 = np.argmax(vs_prob_3,1)

        dag_prob_3 = (best_weight*dag_prob_1 + (1-best_weight)*dag_prob_2)
        dag_pred_3 = np.argmax(dag_prob_3,1)

        pn_acc = np.mean(pn_pred_3==test_img_pn_label)
        str_acc = np.mean(str_pred_3==test_img_str_label)
        pig_acc = np.mean(pig_pred_3==test_img_pig_label)
        rs_acc = np.mean(rs_pred_3==test_img_rs_label)
        dag_acc = np.mean(dag_pred_3==test_img_dag_label)
        bwv_acc = np.mean(bwv_pred_3==test_img_bwv_label)
        vs_acc = np.mean(vs_pred_3==test_img_vs_label)
        diag_acc = np.mean(pred_3==test_img_diag_label)

        avg_acc = (pn_acc + str_acc + pig_acc + rs_acc + dag_acc + bwv_acc + vs_acc + diag_acc) / 8
        mean_avg_acc_list.append(avg_acc)
        p3_list  = [prob_3,pn_prob_3,str_prob_3,pig_prob_3,rs_prob_3,dag_prob_3,bwv_prob_3,vs_prob_3]
        plot_dir_p3 = './{}_{}_{}_{}_{}_p3_plot_figure/{}/'.format(model_name, mode,
                                                                      classifier_name, p1_p2_fused_mode, candidate_mode,
                                                                         i)
        save_gt_result(plot_dir_p3,test_label_list,p3_list)
        print('p1_p2_fused_mode : {}'.format(p1_p2_fused_mode))
        print('P3--------------------------')
        print('avg_acc : {}'.format(avg_acc))
        print('vs_acc : {}'.format(vs_acc))
        print('bwv_acc : {}'.format(bwv_acc))
        print('dag_acc : {}'.format(dag_acc))
        print('rs_acc : {}'.format(rs_acc))
        print('pig_acc : {}'.format(pig_acc))
        print('str_acc : {}'.format(str_acc))
        print('pn_acc : {}'.format(pn_acc))
        print('diag_acc : {}'.format(diag_acc))

averaged_avg_acc = np.mean([mean_avg_acc_list[0],mean_avg_acc_list[2],mean_avg_acc_list[4],mean_avg_acc_list[6],mean_avg_acc_list[8]])
searched_avg_acc = np.mean([mean_avg_acc_list[1],mean_avg_acc_list[3],mean_avg_acc_list[5],mean_avg_acc_list[7],mean_avg_acc_list[9]])

print('averaged avg acc: {}'.format(averaged_avg_acc))
print('searched avg acc: {}'.format(searched_avg_acc))


# %%
#hyperparameters
n_features = 20
search_model = ['FusionM4Net-FS']
mode = 'multimodal'
model_name  = 'FusionM4Net-FS'
search_num = 25
TTA = 6

candidate_mode = 'Linear'
size = 224
round_ = 5
data_mode = 'Normal'
#for i in range(round_):
#i = 55
classifier_name = 'SVM'
p1_p2_fused_mode = 'searched'

p1_acc_list = []
p2_acc_list = []
p3_acc_list = []

mean_avg_acc_list = []

############################get test dataset predictions
print('predicting on testing dataset....P_1')
for j in range(round_):
    i =  100+j
    #weight_file = './{}_{}_{}_weight_file/{}/checkpoint/best_mean_acc_model.pth'.format(mode,model_name,data_mode,j)
    weight_file = './weight/{}/checkpoint/best_mean_acc_model.pth'.format(j)

    print(weight_file)
    net = FusionNet(class_list).cuda()
    out_dir = './{}_{}_result/{}/'.format(mode,model_name, i)
    net.load_state_dict(torch.load(weight_file))

    plot_dir_p1 = './{}_{}_{}_p1_plot_figure/{}/'.format(mode, model_name,candidate_mode, i)

    if model_name in search_model:
        acc_array, weight_array, weight_list  = find_best_threshold(net,val_index_list,df,weight_file,
                                                                    model_name,out_dir,mode,search_num,TTA,size,candidate_mode)

    (avg_acc,
     [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob], 
     [nevu_label, bcc_label, mel_label, misc_label, sk_label],
     [nevu_prob ,bcc_prob ,  mel_prob ,misc_prob ,sk_prob],
     seven_point_feature_list,
     [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt] )= predict(net,test_index_list,df,weight_file,
                                                                    model_name,out_dir,mode,weight_list,TTA,size)

    print(i,weight_list)
    np.save('weight_list_{}_{}.npy'.format(candidate_mode,i),np.array(weight_list))
    p1_acc_list.append(avg_acc)
    prob_1 = prob
    pn_prob_1 = pn_prob
    str_prob_1 = str_prob
    pig_prob_1 = pig_prob
    rs_prob_1  = rs_prob
    dag_prob_1 = dag_prob
    bwv_prob_1 = bwv_prob
    vs_prob_1  = vs_prob
    seven_point_feature_list_p1 = seven_point_feature_list

    test_label_list = [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt]
    p1_list  = [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob]

    save_gt_result(plot_dir_p1,test_label_list,p1_list)
    print('Done!')

    #################get validation dataset predictions#####################
    print('predicting on validation dataset....P_val')


    (avg_acc,
     [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob], 
     [nevu_label, bcc_label, mel_label, misc_label, sk_label],
     [nevu_prob ,bcc_prob ,  mel_prob ,misc_prob ,sk_prob],
     seven_point_feature_list,
     [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt] )= predict(net,val_index_list,df,weight_file,
                                                                    model_name,out_dir,mode,weight_list,TTA,size)

    prob_val = prob
    pn_prob_val = pn_prob
    str_prob_val = str_prob
    pig_prob_val = pig_prob
    rs_prob_val  = rs_prob
    dag_prob_val = dag_prob
    bwv_prob_val = bwv_prob
    vs_prob_val  = vs_prob

    seven_point_feature_list_val = seven_point_feature_list
    print('Done!')

    #################get sub-test (divided from train) dataset predictions#####################
    print('predicting on sub-train (divided from train dataset) dataset....P_train')
    weight_file_evaluated = './{}_{}_{}_weight_file/{}/checkpoint/best_mean_acc_model.pth'.format(mode,model_name,'self_evaluated',
                                                                                       i)
    weight_file_evaluated = './self_evaluated/{}/checkpoint/best_mean_acc_model.pth'.format(j)
    out_dir = './{}_{}_{}_result/{}/'.format(mode, model_name,candidate_mode, i)

    net.load_state_dict(torch.load(weight_file_evaluated))

    (avg_acc,
     [prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob], 
     [nevu_label, bcc_label, mel_label, misc_label, sk_label],
     [nevu_prob ,bcc_prob ,  mel_prob ,misc_prob ,sk_prob],
     seven_point_feature_list,
     [gt,pn_gt,str_gt,pig_gt,rs_gt,dag_gt, bwv_gt,vs_gt] )= predict(net,train_index_list_2,df,weight_file,
                                                                    model_name,out_dir,mode,weight_list,TTA,size)



    prob_self_evaluated = prob
    pn_prob_self_evaluated = pn_prob
    str_prob_self_evaluated = str_prob
    pig_prob_self_evaluated = pig_prob
    rs_prob_self_evaluated  = rs_prob
    dag_prob_self_evaluated = dag_prob
    bwv_prob_self_evaluated = bwv_prob
    vs_prob_self_evaluated  = vs_prob

    seven_point_feature_list_self_evaluated = seven_point_feature_list


    #################get labels#####################
    print('fusing fusing predictions from two-modality images and metadata......')
    train_label_dict,train_meta_list = get_label_list(train_index_list_2)
    val_label_dict,val_meta_list   = get_label_list(val_index_list)
    test_label_dict,test_meta_list  = get_label_list(test_index_list)



    #################fusing predictions from two-modality images and metadata by concatenation#####################
    train_meta_array = np.array(train_meta_list)
    img_total_feature = np.concatenate([
                                  seven_point_feature_list_self_evaluated,
                                  train_meta_array      
                                  ],1)
    print(img_total_feature.shape)
    train_diag_label = np.array(train_label_dict['diag'])
    print(train_diag_label.shape)

    val_meta_array = np.array(val_meta_list)
    val_img_total_feature = np.concatenate([
                                 seven_point_feature_list_val,
                                 val_meta_array      
                                  ],1)

    print(val_img_total_feature.shape)
    val_diag_label = np.array(val_label_dict['diag'])
    print(val_diag_label.shape)

    test_meta_array = np.array(test_meta_list)
    test_img_total_feature = np.concatenate([
                                  seven_point_feature_list_p1,
                                  test_meta_array      
                                  ],1)
    
    print(test_img_total_feature.shape)
    test_diag_label = np.array(test_label_dict['diag'])
    print(test_diag_label.shape)
    print('Done!')

    img_pn_label= np.array(train_label_dict['pn'])
    img_str_label= np.array(train_label_dict['str'])
    img_pig_label= np.array(train_label_dict['pig'])
    img_rs_label= np.array(train_label_dict['rs'])
    img_dag_label= np.array(train_label_dict['dag'])
    img_bwv_label= np.array(train_label_dict['bwv'])
    img_vs_label= np.array(train_label_dict['vs'])
    img_diag_label = np.array(train_label_dict['diag'])

    test_img_pn_label= np.array(test_label_dict['pn'])
    test_img_str_label= np.array(test_label_dict['str'])
    test_img_pig_label= np.array(test_label_dict['pig'])
    test_img_rs_label= np.array(test_label_dict['rs'])
    test_img_dag_label= np.array(test_label_dict['dag'])
    test_img_bwv_label= np.array(test_label_dict['bwv'])
    test_img_vs_label= np.array(test_label_dict['vs'])
    test_img_diag_label = np.array(test_label_dict['diag'])


    val_img_pn_label= np.array(val_label_dict['pn'])
    val_img_str_label= np.array(val_label_dict['str'])
    val_img_pig_label= np.array(val_label_dict['pig'])
    val_img_rs_label= np.array(val_label_dict['rs'])
    val_img_dag_label= np.array(val_label_dict['dag'])
    val_img_bwv_label= np.array(val_label_dict['bwv'])
    val_img_vs_label= np.array(val_label_dict['vs'])
    val_img_diag_label= np.array(val_label_dict['diag'])


    img_pn_label_one_hot = to_categorical(np.array(train_label_dict['pn']))
    img_str_label_one_hot= to_categorical(np.array(train_label_dict['str']))
    img_pig_labe_one_hotl= to_categorical(np.array(train_label_dict['pig']))
    img_rs_label_one_hot= to_categorical(np.array(train_label_dict['rs']))
    img_dag_label_one_hot= to_categorical(np.array(train_label_dict['dag']))
    img_bwv_label_one_hot= to_categorical(np.array(train_label_dict['bwv']))
    img_vs_label_one_hot= to_categorical(np.array(train_label_dict['vs']))

    test_img_pn_label_one_hot = to_categorical(np.array(test_label_dict['pn']))
    test_img_str_label_one_hot= to_categorical(np.array(test_label_dict['str']))
    test_img_pig_labe_one_hotl= to_categorical(np.array(test_label_dict['pig']))
    test_img_rs_label_one_hot= to_categorical(np.array(test_label_dict['rs']))
    test_img_dag_label_one_hot= to_categorical(np.array(test_label_dict['dag']))
    test_img_bwv_label_one_hot= to_categorical(np.array(test_label_dict['bwv']))
    test_img_vs_label_one_hot= to_categorical(np.array(test_label_dict['vs']))

    print('Training Classifier Cluster..... Classifier：{}'.format(classifier_name))
    _,[prob_2,pn_prob_2,str_prob_2,pig_prob_2,rs_prob_2,dag_prob_2,bwv_prob_2,vs_prob_2] = classifier_cluster_predict(test_img_total_feature,classifier_name)
    _,[val_p2_prob,val_p2_pn_prob,val_p2_str_prob,val_p2_pig_prob,val_p2_rs_prob,val_p2_dag_prob,val_p2_bwv_prob,val_p2_vs_prob] = classifier_cluster_predict(val_img_total_feature,classifier_name)
    print('Done!')
    p2_list  = [prob_2,pn_prob_2,str_prob_2,pig_prob_2,rs_prob_2,dag_prob_2,bwv_prob_2,vs_prob_2]
    plot_dir_p2 = './{}_{}_{}_p2_plot_figure/{}/'.format( mode,model_name, 
                                                                  candidate_mode, i)
    save_gt_result(plot_dir_p2,test_label_list,p2_list)


    pn_test_preds = np.argmax(pn_prob_2,1)
    str_test_preds = np.argmax(str_prob_2,1)
    pig_test_preds = np.argmax(pig_prob_2,1)
    rs_test_preds = np.argmax(rs_prob_2,1)
    dag_test_preds = np.argmax(dag_prob_2,1)
    bwv_test_preds = np.argmax(bwv_prob_2,1)
    vs_test_preds = np.argmax(vs_prob_2,1)
    diag_test_preds = np.argmax(prob_2,1)

    pn_acc = np.mean(pn_test_preds==test_img_pn_label)
    str_acc = np.mean(str_test_preds==test_img_str_label)
    pig_acc = np.mean(pig_test_preds==test_img_pig_label)
    rs_acc = np.mean(rs_test_preds==test_img_rs_label)
    dag_acc = np.mean(dag_test_preds==test_img_dag_label)
    bwv_acc = np.mean(bwv_test_preds==test_img_bwv_label)
    vs_acc = np.mean(vs_test_preds==test_img_vs_label)
    diag_acc = np.mean(diag_test_preds==test_img_diag_label)
    avg_acc = (pn_acc + str_acc + pig_acc + rs_acc + dag_acc + bwv_acc + vs_acc + diag_acc) / 8

    print('P2--------------------------')
    print('avg_acc : {}'.format(avg_acc))
    print('vs_acc : {}'.format(vs_acc))
    print('bwv_acc : {}'.format(bwv_acc))
    print('dag_acc : {}'.format(dag_acc))
    print('rs_acc : {}'.format(rs_acc))
    print('pig_acc : {}'.format(pig_acc))
    print('str_acc : {}'.format(str_acc))
    print('pn_acc : {}'.format(pn_acc))
    print('diag_acc : {}'.format(diag_acc))



    vs_weight_list,vs_acc_list,vs_best_weight,vs_best_acc = prediction_2_weight_search(vs_prob_val,val_p2_vs_prob,val_img_vs_label)
    str_weight_list,str_acc_list,str_best_weight,str_best_acc = prediction_2_weight_search(str_prob_val,val_p2_str_prob,val_img_str_label)
    pn_weight_list,pn_acc_list,pn_best_weight,pn_best_acc = prediction_2_weight_search(pn_prob_val,val_p2_pn_prob,val_img_pn_label)
    pig_weight_list,pig_acc_list,pig_best_weight,pig_best_acc = prediction_2_weight_search(pig_prob_val,val_p2_pig_prob,val_img_pig_label)
    bwv_weight_list,bwv_acc_list,bwv_best_weight,bwv_best_acc = prediction_2_weight_search(bwv_prob_val,val_p2_bwv_prob,val_img_bwv_label)
    dag_weight_list,dag_acc_list,dag_best_weight,dag_best_acc = prediction_2_weight_search(dag_prob_val,val_p2_dag_prob,val_img_dag_label)
    rs_weight_list,rs_acc_list,rs_best_weight,rs_best_acc = prediction_2_weight_search(rs_prob_val,val_p2_rs_prob,val_img_rs_label)
    pred_weight_list,pred_acc_list,pred_best_weight,pred_best_acc = prediction_2_weight_search(prob_val,val_p2_prob,val_img_diag_label)

    vs_arr = np.array(vs_acc_list)
    str_arr = np.array(str_acc_list)
    pn_arr = np.array(pn_acc_list)
    pig_arr = np.array(pig_acc_list)
    bwv_arr = np.array(bwv_acc_list)
    dag_arr = np.array(dag_acc_list)
    rs_arr = np.array(rs_acc_list)
    pred_arr = np.array(pred_acc_list)

    avg_arr = (vs_arr+str_arr+pn_arr+pig_arr+bwv_arr+dag_arr+rs_arr+pred_arr)/8
    index = np.argmax(avg_arr)

    print('searched P1-P2 fused weights----')
    print(pred_weight_list[index])
    print(avg_arr[index])

    best_weight = pred_weight_list[index]
    p1_p2_weight_list = [0.5,best_weight]
    p1_p2_fused_mode_list = ['average','searched']

    for index in range(len(p1_p2_weight_list)):

        best_weight = p1_p2_weight_list[index]
        p1_p2_fused_mode = p1_p2_fused_mode_list[index]

        prob_3 = (best_weight*prob_1+(1-best_weight)*prob_2)
        pred_3 = np.argmax(prob_3,1)

        pn_prob_3 = (best_weight*pn_prob_1 + (1-best_weight)*pn_prob_2)
        pn_pred_3 = np.argmax(pn_prob_3,1)

        pig_prob_3 = (best_weight*pig_prob_1 +(1-best_weight)*pig_prob_2)
        pig_pred_3 = np.argmax(pig_prob_3,1)

        str_prob_3 = (best_weight*str_prob_1 + (1-best_weight)*str_prob_2)
        str_pred_3 = np.argmax(str_prob_3,1)

        rs_prob_3 = (best_weight*rs_prob_1 + (1-best_weight)*rs_prob_2)
        rs_pred_3 = np.argmax(rs_prob_3,1)

        bwv_prob_3 = (best_weight*bwv_prob_1 + (1-best_weight)*bwv_prob_2)
        bwv_pred_3 = np.argmax(bwv_prob_3,1)

        vs_prob_3 = (best_weight*vs_prob_1 + (1-best_weight)*vs_prob_2)
        vs_pred_3 = np.argmax(vs_prob_3,1)

        dag_prob_3 = (best_weight*dag_prob_1 + (1-best_weight)*dag_prob_2)
        dag_pred_3 = np.argmax(dag_prob_3,1)

        pn_acc = np.mean(pn_pred_3==test_img_pn_label)
        str_acc = np.mean(str_pred_3==test_img_str_label)
        pig_acc = np.mean(pig_pred_3==test_img_pig_label)
        rs_acc = np.mean(rs_pred_3==test_img_rs_label)
        dag_acc = np.mean(dag_pred_3==test_img_dag_label)
        bwv_acc = np.mean(bwv_pred_3==test_img_bwv_label)
        vs_acc = np.mean(vs_pred_3==test_img_vs_label)
        diag_acc = np.mean(pred_3==test_img_diag_label)

        avg_acc = (pn_acc + str_acc + pig_acc + rs_acc + dag_acc + bwv_acc + vs_acc + diag_acc) / 8
        mean_avg_acc_list.append(avg_acc)
        p3_list  = [prob_3,pn_prob_3,str_prob_3,pig_prob_3,rs_prob_3,dag_prob_3,bwv_prob_3,vs_prob_3]
        plot_dir_p3 = './{}_{}_{}_{}_{}_p3_plot_figure/{}/'.format(model_name, mode,
                                                                      classifier_name, p1_p2_fused_mode, candidate_mode,
                                                                         i)
        save_gt_result(plot_dir_p3,test_label_list,p3_list)
        print('p1_p2_fused_mode : {}'.format(p1_p2_fused_mode))
        print('P3--------------------------')
        print('avg_acc : {}'.format(avg_acc))
        print('vs_acc : {}'.format(vs_acc))
        print('bwv_acc : {}'.format(bwv_acc))
        print('dag_acc : {}'.format(dag_acc))
        print('rs_acc : {}'.format(rs_acc))
        print('pig_acc : {}'.format(pig_acc))
        print('str_acc : {}'.format(str_acc))
        print('pn_acc : {}'.format(pn_acc))
        print('diag_acc : {}'.format(diag_acc))

averaged_avg_acc = np.mean([mean_avg_acc_list[0],mean_avg_acc_list[2],mean_avg_acc_list[4],mean_avg_acc_list[6],mean_avg_acc_list[8]])
searched_avg_acc = np.mean([mean_avg_acc_list[1],mean_avg_acc_list[3],mean_avg_acc_list[5],mean_avg_acc_list[7],mean_avg_acc_list[9]])

print('averaged avg acc: {}'.format(averaged_avg_acc))
print('searched avg acc: {}'.format(searched_avg_acc))


# %%



