import argparse
from torch.utils.data import DataLoader
import pandas as pd
from fusionm4net_dependency import *
from torch.utils import data
import numpy as np
import cv2
from keras.utils import to_categorical
import torch



def encode_meta_label(img_info,index_num):

    level_of_diagnostic_difficulty_label = img_info['level_of_diagnostic_difficulty'][index_num]
    #print(level_of_diagnostic_difficulty_label)
    for index,label in enumerate(level_of_diagnostic_difficulty_label_list):

        if level_of_diagnostic_difficulty_label in label:
            level_of_diagnostic_difficulty_index = index
            level_of_diagnostic_difficulty_label_one_hot = to_categorical(level_of_diagnostic_difficulty_index,num_level_of_diagnostic_difficulty_label_list)
        else:
            continue

    evaluation_label = img_info['elevation'][index_num]
    for index,label in enumerate(evaluation_list):
        if evaluation_label in label:
            evaluation_label_index = index
            evaluation_label_one_hot = to_categorical(evaluation_label_index,num_evaluation_list)
        else:
            continue

    sex_label = img_info['sex'][index_num]
    for index,label in enumerate(sex_list):
        if sex_label in label:
            sex_label_index = index
            sex_label_one_hot = to_categorical(sex_label_index,num_sex_list)
        else:
            continue

    location_label = img_info['location'][index_num]
    for index,label in enumerate(location_list):
        if location_label in label:
            location_label_index = index
            location_label_one_hot = to_categorical(location_label_index,num_location_list)
        else:
            continue

    management_label = img_info['management'][index_num]
    for index,label in enumerate(management_list):
        if management_label in label:
            management_label_index = index
            management_label_one_hot = to_categorical(management_label_index,num_management_list)
        else:
            continue

    meta_vector = np.hstack([
    level_of_diagnostic_difficulty_label_one_hot,
    evaluation_label_one_hot,
    location_label_one_hot,
    sex_label_one_hot,
    management_label_one_hot
    ])
    
    

    return meta_vector, None, None



def encode_label(img_info, index_num):
    # Encode the diagnositic label
    diagnosis_label = img_info['diagnosis'][index_num]
    for index, label in enumerate(label_list):
        if diagnosis_label in label:
            diagnosis_index = index
            diagnosis_label_one_hot = to_categorical(diagnosis_index, num_label)
        # print(index_num,diagnosis_index,diagnosis_label,diagnosis_label_one_hot)
        else:
            continue
    #Encode the Seven-point label
    # 1
    pigment_network_label = img_info['pigment_network'][index_num]
    for index, label in enumerate(pigment_network_label_list):
        if pigment_network_label in label:
            pigment_network_index = index
            pigment_network_label_one_hot = to_categorical(pigment_network_index, num_pigment_network_label)
        else:
            continue
    # 2
    streaks_label = img_info['streaks'][index_num]
    for index, label in enumerate(streaks_label_list):
        if streaks_label in label:
            streaks_index = index
            streaks_label_one_hot = to_categorical(streaks_index, num_streaks_label)
        else:
            continue
    # 3
    pigmentation_label = img_info['pigmentation'][index_num]
    for index, label in enumerate(pigmentation_label_list):
        if pigmentation_label in label:
            pigmentation_index = index
            pigmentation_label_one_hot = to_categorical(pigmentation_index, num_pigmentation_label)
        else:
            continue
    # 4
    regression_structures_label = img_info['regression_structures'][index_num]
    for index, label in enumerate(regression_structures_label_list):
        if regression_structures_label in label:
            regression_structures_index = index
            regression_structures_label_one_hot = to_categorical(regression_structures_index,
                                                                 num_regression_structures_label)
        else:
            continue
    # 5
    dots_and_globules_label = img_info['dots_and_globules'][index_num]
    for index, label in enumerate(dots_and_globules_label_list):
        if dots_and_globules_label in label:
            dots_and_globules_index = index
            dots_and_globules_label_one_hot = to_categorical(dots_and_globules_index, num_dots_and_globules_label)
        else:
            continue
    # 6
    blue_whitish_veil_label = img_info['blue_whitish_veil'][index_num]
    for index, label in enumerate(blue_whitish_veil_label_list):
        if blue_whitish_veil_label in label:
            blue_whitish_veil_index = index
            blue_whitish_veil_label_one_hot = to_categorical(blue_whitish_veil_index, num_blue_whitish_veil_label)
        else:
            continue
    # 7
    vascular_structures_label = img_info['vascular_structures'][index_num]
    for index, label in enumerate(vascular_structures_label_list):
        if vascular_structures_label in label:
            vascular_structures_index = index
            vascular_structures_label_one_hot = to_categorical(vascular_structures_index, num_vascular_structures_label)
        else:
            continue

    return np.array([diagnosis_index,
                     pigment_network_index,
                     streaks_index,
                     pigmentation_index,
                     regression_structures_index,
                     dots_and_globules_index,
                     blue_whitish_veil_index,
                     vascular_structures_index])



def load_image(path, shape):
    img = cv2.imread(path)
    img = cv2.resize(img, (shape[0], shape[1]))

    return img


class SkinDataset(data.Dataset):
    def __init__(self, image_dir, img_info, file_list, shape, is_test=False, num_class=1):
        self.is_test = is_test
        self.image_dir = image_dir
        self.img_info = img_info
        self.file_list = file_list
        self.shape = shape
        self.num_class = num_class
        self.total_img_info = img_info
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]
        sub_img_info = self.total_img_info[file_id:file_id + 1]
                
        # get the clincal image path
        clinic_img_path = sub_img_info['clinic']
        # get the dermoscopy image path
        dermoscopy_img_path = sub_img_info['derm']
        # load the clinical image
        clinic_img = load_image(self.image_dir + clinic_img_path[file_id], self.shape)
        # load the dermoscopy image
        dermoscopy_img = load_image(self.image_dir + dermoscopy_img_path[file_id], self.shape)

        # Encode the diagnositic label
        diagnosis_label = sub_img_info['diagnosis'][file_id]
        for index_label, label in enumerate(label_list):
            if diagnosis_label in label:
                diagnosis_index = index_label
                diagnosis_label_one_hot = to_categorical(diagnosis_index, num_label)
            else:
                continue

        if not self.is_test:
            augmented = aug(image=clinic_img, mask=dermoscopy_img)
            clinic_img = augmented['image']
            dermoscopy_img = augmented['mask']

        total_label = encode_label(sub_img_info, file_id)
        # print(total_label)
        clinic_img = torch.from_numpy(np.transpose(clinic_img, (2, 0, 1)).astype('float32') / 255)
        dermoscopy_img = torch.from_numpy(np.transpose(dermoscopy_img, (2, 0, 1)).astype('float32') / 255)
        meta_data, _, _ = encode_meta_label(sub_img_info, file_id)

        return clinic_img, dermoscopy_img,  meta_data, [total_label[0], total_label[1], total_label[2], total_label[3],
                                        total_label[4], total_label[5], total_label[6], total_label[7]]
    


def generate_dataloader(shape,batch_size,num_workers):
    test_index_df = pd.read_csv(test_index_path)
    train_index_df = pd.read_csv(train_index_path)
    val_index_df = pd.read_csv(val_index_path)

    train_index_list = list(train_index_df['indexes'])
    val_index_list = list(val_index_df['indexes'])
    test_index_list = list(test_index_df['indexes'])

    train_index_list_1 = train_index_list[0:206]
    train_index_list_2 = train_index_list[206:]


    df = pd.read_csv(img_info_path)
    # if data_mode == 'self_evaluated':
    #   data_mode = 'SP'
    #   train_skindataset = SkinDataset(image_dir=source_dir,
    #                                   img_info=df,
    #                                   file_list=train_index_list_1,
    #                                   shape=shape, is_test=False,
    #                                   )
    #   train_dataloader = DataLoader(
    #       dataset=train_skindataset,
    #       batch_size=batch_size,
    #       num_workers=num_workers,
    #       pin_memory=True,
    #       shuffle=True)
  
    #   val_skindataset = SkinDataset(image_dir=source_dir,
    #                                 img_info=df,
    #                                 file_list=val_index_list,
    #                                 shape=shape, is_test=True,
    #                                 )
  
    #   val_dataloader = DataLoader(
    #       dataset=val_skindataset,
    #       batch_size=batch_size,
    #       num_workers=num_workers,
    #       pin_memory=True,
    #       shuffle=True)
    if False:
        pass
    else:
        train_skindataset = SkinDataset(image_dir=source_dir,
                                        img_info=df,
                                        file_list=train_index_list,
                                        shape=shape, 
                                        is_test=False,
                                        )
        train_dataloader = DataLoader(
            dataset=train_skindataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True)

        val_skindataset = SkinDataset(image_dir=source_dir,
                                    img_info=df,
                                    file_list=val_index_list,
                                    shape=shape, 
                                    is_test=True,
                                    )

        val_dataloader = DataLoader(
            dataset=val_skindataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True)

    return train_dataloader, val_dataloader



if __name__ == '__main__':
    # Hyperparameters

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_mode', type=str, default='Normal', help='Normal or self_evaluated', choices=['Normal', 'self_evaluated'])
    args = parser.parse_args()
    
    mode = 'multimodal'
    model_name = 'FusionM4Net-FS'
    shape = (224, 224)
    batch_size = 32
    num_workers = 8
    data_mode = args.data_mode
    if(data_mode != 'Normal'):
        raise NotImplementedError()
    deterministic = True
    if deterministic:
        if data_mode == 'Normal':
          random_seeds = 170
        # elif data_mode == 'self_evaluated':
        #   random_seeds = 183
    rounds = 1
    lr = 3e-5
    epochs = 250
    swa_epoch = 50

    train_dataloader, val_dataloader = generate_dataloader(shape, batch_size, num_workers)
    print("Done")
    
    for i in range(rounds):
        if deterministic:
            set_seed(random_seeds + i)
      # create logger
        print(random_seeds+i)
        log, out_dir = CraateLogger(mode, model_name,i,data_mode)
        net = FusionNet(class_list).cuda()
      # create optimizer
        optimizer = optim.Adam(net.parameters(), lr=lr)
        opt = SWA(optimizer)
      # create learning schdule
        cosine_learning_schule = create_cosine_learing_schdule(epochs, lr)
        run_train(model_name,mode,i)




