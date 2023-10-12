import os
import logging
import torch


def train(net, train_dataloader, optimizer, device):

    net.set_mode('train')
    train_loss = 0
    train_dia_acc = 0
    train_sps_acc = 0
    for index, (clinic_image, derm_image, meta_data, label) in enumerate(train_dataloader):
        optimizer.zero_grad()

        clinic_image = clinic_image.to(device)
        derm_image = derm_image.to(device)
        meta_data = meta_data.to(device)

        # Diagostic label
        diagnosis_label = label[0].long().to(device)
        # Seven-Point Checklikst labels
        pn_label = label[1].long().to(device)
        str_label = label[2].long().to(device)
        pig_label = label[3].long().to(device)
        rs_label = label[4].long().to(device)
        dag_label = label[5].long().to(device)
        bwv_label = label[6].long().to(device)
        vs_label = label[7].long().to(device)

        [(logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm,
          logit_vs_derm),
         (logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic, logit_bwv_clic,
          logit_vs_clic),
         (logit_diagnosis_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion,
          logit_vs_fusion)] = net((clinic_image, derm_image))

        # average fusion loss
        loss_fusion = torch.true_divide(
            net.criterion(logit_diagnosis_fusion, diagnosis_label)
            + net.criterion(logit_pn_fusion, pn_label)
            + net.criterion(logit_str_fusion, str_label)
            + net.criterion(logit_pig_fusion, pig_label)
            + net.criterion(logit_rs_fusion, rs_label)
            + net.criterion(logit_dag_fusion, dag_label)
            + net.criterion(logit_bwv_fusion, bwv_label)
            + net.criterion(logit_vs_fusion, vs_label), 8)

        # average clic loss
        loss_clic = torch.true_divide(
            net.criterion(logit_diagnosis_clic, diagnosis_label)
            + net.criterion(logit_pn_clic, pn_label)
            + net.criterion(logit_str_clic, str_label)
            + net.criterion(logit_pig_clic, pig_label)
            + net.criterion(logit_rs_clic, rs_label)
            + net.criterion(logit_dag_clic, dag_label)
            + net.criterion(logit_bwv_clic, bwv_label)
            + net.criterion(logit_vs_clic, vs_label), 8)

        # average derm loss
        loss_derm = torch.true_divide(
            net.criterion(logit_diagnosis_derm, diagnosis_label)
            + net.criterion(logit_pn_derm, pn_label)
            + net.criterion(logit_str_derm, str_label)
            + net.criterion(logit_pig_derm, pig_label)
            + net.criterion(logit_rs_derm, rs_label)
            + net.criterion(logit_dag_derm, dag_label)
            + net.criterion(logit_bwv_derm, bwv_label)
            + net.criterion(logit_vs_derm, vs_label), 8)

        # average loss
        loss = loss_fusion*0.33 + loss_clic*0.33 + loss_derm*0.33

        # fusion, clic, derm accuracy for diagnostic
        dia_acc_fusion = torch.true_divide(net.metric(
            logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))
        dia_acc_clic = torch.true_divide(net.metric(
            logit_diagnosis_clic, diagnosis_label), clinic_image.size(0))
        dia_acc_derm = torch.true_divide(net.metric(
            logit_diagnosis_derm, diagnosis_label), clinic_image.size(0))

        # average accuracy for diagnostic
        dia_acc = torch.true_divide(
            dia_acc_fusion + dia_acc_clic + dia_acc_derm, 3)

        # seven-point accuracy of fusion
        sps_acc_fusion = torch.true_divide(net.metric(logit_pn_fusion, pn_label)
                                           + net.metric(logit_str_fusion, str_label)
                                           + net.metric(logit_pig_fusion, pig_label)
                                           + net.metric(logit_rs_fusion, rs_label)
                                           + net.metric(logit_dag_fusion, dag_label)
                                           + net.metric(logit_bwv_fusion, bwv_label)
                                           + net.metric(logit_vs_fusion, vs_label), 
                                                7 * clinic_image.size(0))

        # seven-point accuracy of clic
        sps_acc_clic = torch.true_divide(net.metric(logit_pn_clic, pn_label)
                                         + net.metric(logit_str_clic, str_label)
                                         + net.metric(logit_pig_clic, pig_label)
                                         + net.metric(logit_rs_clic, rs_label)
                                         + net.metric(logit_dag_clic, dag_label)
                                         + net.metric(logit_bwv_clic, bwv_label)
                                         + net.metric(logit_vs_clic, vs_label), 
                                                7 * clinic_image.size(0))
        # seven-point accuracy of derm
        sps_acc_derm = torch.true_divide(net.metric(logit_pn_derm, pn_label)
                                         + net.metric(logit_str_derm, str_label)
                                         + net.metric(logit_pig_derm, pig_label)
                                         + net.metric(logit_rs_derm, rs_label)
                                         + net.metric(logit_dag_derm, dag_label)
                                         + net.metric(logit_bwv_derm, bwv_label)
                                         + net.metric(logit_vs_derm, vs_label), 
                                                7 * clinic_image.size(0))
        # average seven-point accuracy
        sps_acc = torch.true_divide(sps_acc_fusion + sps_acc_clic + sps_acc_derm, 3)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dia_acc += dia_acc.item()
        train_sps_acc += sps_acc.item()

    # Because the index start with the value 0f zero
    train_loss = train_loss / (index + 1)
    train_dia_acc = train_dia_acc / (index + 1)
    train_sps_acc = train_sps_acc / (index + 1)

    return train_loss, train_dia_acc, train_sps_acc


def validation(net, val_dataloader, device):
    net.set_mode('valid')
    val_loss = 0
    val_dia_acc = 0
    val_sps_acc = 0
    for index, (clinic_image, derm_image, meta_data, label) in enumerate(val_dataloader):

        clinic_image = clinic_image.to(device)
        derm_image = derm_image.to(device)
        meta_data = meta_data.to(device)

        diagnosis_label = label[0].long().to(device)
        pn_label = label[1].long().to(device)
        str_label = label[2].long().to(device)
        pig_label = label[3].long().to(device)
        rs_label = label[4].long().to(device)
        dag_label = label[5].long().to(device)
        bwv_label = label[6].long().to(device)
        vs_label = label[7].long().to(device)

        with torch.no_grad():

            [(logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm,
                logit_vs_derm),
                (logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic, logit_bwv_clic,
                 logit_vs_clic),
                (logit_diagnosis_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion,
                 logit_vs_fusion)] = net((clinic_image, derm_image))

            loss_fusion = torch.true_divide(
                net.criterion(logit_diagnosis_fusion, diagnosis_label)
                + net.criterion(logit_pn_fusion, pn_label)
                + net.criterion(logit_str_fusion, str_label)
                + net.criterion(logit_pig_fusion, pig_label)
                + net.criterion(logit_rs_fusion, rs_label)
                + net.criterion(logit_dag_fusion, dag_label)
                + net.criterion(logit_bwv_fusion, bwv_label)
                + net.criterion(logit_vs_fusion, vs_label), 8)

            loss_clic = torch.true_divide(
                net.criterion(logit_diagnosis_clic, diagnosis_label)
                + net.criterion(logit_pn_clic, pn_label)
                + net.criterion(logit_str_clic, str_label)
                + net.criterion(logit_pig_clic, pig_label)
                + net.criterion(logit_rs_clic, rs_label)
                + net.criterion(logit_dag_clic, dag_label)
                + net.criterion(logit_bwv_clic, bwv_label)
                + net.criterion(logit_vs_clic, vs_label), 8)

            loss_derm = torch.true_divide(
                net.criterion(logit_diagnosis_derm, diagnosis_label)
                + net.criterion(logit_pn_derm, pn_label)
                + net.criterion(logit_str_derm, str_label)
                + net.criterion(logit_pig_derm, pig_label)
                + net.criterion(logit_rs_derm, rs_label)
                + net.criterion(logit_dag_derm, dag_label)
                + net.criterion(logit_bwv_derm, bwv_label)
                + net.criterion(logit_vs_derm, vs_label), 8)

            loss = loss_fusion*0.33 + loss_clic*0.33 + loss_derm*0.33

            dia_acc_fusion = torch.true_divide(net.metric(
                logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))
            dia_acc_clic = torch.true_divide(net.metric(
                logit_diagnosis_clic, diagnosis_label), clinic_image.size(0))
            dia_acc_derm = torch.true_divide(net.metric(
                logit_diagnosis_derm, diagnosis_label), clinic_image.size(0))

            dia_acc = torch.true_divide(
                dia_acc_fusion + dia_acc_clic + dia_acc_derm, 3)

            sps_acc_fusion = torch.true_divide(net.metric(logit_pn_fusion, pn_label)
                                               + net.metric(logit_str_fusion, str_label)
                                               + net.metric(logit_pig_fusion, pig_label)
                                               + net.metric(logit_rs_fusion, rs_label)
                                               + net.metric(logit_dag_fusion, dag_label)
                                               + net.metric(logit_bwv_fusion, bwv_label)
                                               + net.metric(logit_vs_fusion, vs_label), 7 * clinic_image.size(0))

            sps_acc_clic = torch.true_divide(net.metric(logit_pn_clic, pn_label)
                                             + net.metric(logit_str_clic, str_label)
                                             + net.metric(logit_pig_clic, pig_label)
                                             + net.metric(logit_rs_clic, rs_label)
                                             + net.metric(logit_dag_clic, dag_label)
                                             + net.metric(logit_bwv_clic, bwv_label)
                                             + net.metric(logit_vs_clic, vs_label), 7 * clinic_image.size(0))

            sps_acc_derm = torch.true_divide(net.metric(logit_pn_derm, pn_label)
                                             + net.metric(logit_str_derm, str_label)
                                             + net.metric(logit_pig_derm, pig_label)
                                             + net.metric(logit_rs_derm, rs_label)
                                             + net.metric(logit_dag_derm, dag_label)
                                             + net.metric(logit_bwv_derm, bwv_label)
                                             + net.metric(logit_vs_derm, vs_label), 7 * clinic_image.size(0))

            sps_acc = torch.true_divide(sps_acc_fusion + sps_acc_clic + sps_acc_derm, 3)

        val_loss += loss.item()
        val_dia_acc += dia_acc.item()
        val_sps_acc += sps_acc.item()

    val_loss = val_loss / (index + 1)
    val_dia_acc = val_dia_acc / (index + 1)
    val_sps_acc = val_sps_acc / (index + 1)
    return val_loss, val_dia_acc, val_sps_acc


def set_path(save_root='result', filename_prefix=""):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    save_tag = f"{filename_prefix}split_learning"
    exp_seq_path = os.path.join(save_root, 'exp_seq.txt')

    if not os.path.exists(exp_seq_path):
        file = open(exp_seq_path, 'w')
        exp_seq = 0
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
        os.makedirs(os.path.join(save_path, 'checkpoint'))

    logger_path = os.path.join(save_path, 'exp_log.log')

    return logger_path, exp_seq, save_path


def get_logger(save_root='result', filename_prefix=""):

    logger_path, exp_seq, save_path = set_path(
        save_root=save_root, filename_prefix=filename_prefix)

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

    return logger, exp_seq, save_path
