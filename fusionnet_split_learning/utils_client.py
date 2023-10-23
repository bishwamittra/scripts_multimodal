import torch
from tqdm import tqdm




def validation(client_net, server_net, val_dataloader, device):
    client_net.set_mode('valid')
    server_net.set_mode('valid')
    val_loss = 0
    val_dia_acc = 0
    val_sps_acc = 0
    for (clinic_image, derm_image, meta_data, label) in tqdm(val_dataloader):

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

            
            # Fetch output from client
            output = client_net((clinic_image, derm_image))
            x_clic, x_derm = output
            x_clic = x_clic.clone().detach().requires_grad_(False)
            x_derm = x_derm.clone().detach().requires_grad_(False)

            
            [(logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm,
                logit_vs_derm),
                (logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic, logit_bwv_clic,
                 logit_vs_clic),
                (logit_diagnosis_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion,
                 logit_vs_fusion)] = server_net((x_clic, x_derm))

            loss_fusion = torch.true_divide(
                server_net.criterion(logit_diagnosis_fusion, diagnosis_label)
                + server_net.criterion(logit_pn_fusion, pn_label)
                + server_net.criterion(logit_str_fusion, str_label)
                + server_net.criterion(logit_pig_fusion, pig_label)
                + server_net.criterion(logit_rs_fusion, rs_label)
                + server_net.criterion(logit_dag_fusion, dag_label)
                + server_net.criterion(logit_bwv_fusion, bwv_label)
                + server_net.criterion(logit_vs_fusion, vs_label), 8)

            loss_clic = torch.true_divide(
                server_net.criterion(logit_diagnosis_clic, diagnosis_label)
                + server_net.criterion(logit_pn_clic, pn_label)
                + server_net.criterion(logit_str_clic, str_label)
                + server_net.criterion(logit_pig_clic, pig_label)
                + server_net.criterion(logit_rs_clic, rs_label)
                + server_net.criterion(logit_dag_clic, dag_label)
                + server_net.criterion(logit_bwv_clic, bwv_label)
                + server_net.criterion(logit_vs_clic, vs_label), 8)

            loss_derm = torch.true_divide(
                server_net.criterion(logit_diagnosis_derm, diagnosis_label)
                + server_net.criterion(logit_pn_derm, pn_label)
                + server_net.criterion(logit_str_derm, str_label)
                + server_net.criterion(logit_pig_derm, pig_label)
                + server_net.criterion(logit_rs_derm, rs_label)
                + server_net.criterion(logit_dag_derm, dag_label)
                + server_net.criterion(logit_bwv_derm, bwv_label)
                + server_net.criterion(logit_vs_derm, vs_label), 8)

            loss = loss_fusion*0.33 + loss_clic*0.33 + loss_derm*0.33

            dia_acc_fusion = torch.true_divide(server_net.metric(
                logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))
            dia_acc_clic = torch.true_divide(server_net.metric(
                logit_diagnosis_clic, diagnosis_label), clinic_image.size(0))
            dia_acc_derm = torch.true_divide(server_net.metric(
                logit_diagnosis_derm, diagnosis_label), clinic_image.size(0))

            dia_acc = torch.true_divide(
                dia_acc_fusion + dia_acc_clic + dia_acc_derm, 3)

            sps_acc_fusion = torch.true_divide(server_net.metric(logit_pn_fusion, pn_label)
                                               + server_net.metric(logit_str_fusion, str_label)
                                               + server_net.metric(logit_pig_fusion, pig_label)
                                               + server_net.metric(logit_rs_fusion, rs_label)
                                               + server_net.metric(logit_dag_fusion, dag_label)
                                               + server_net.metric(logit_bwv_fusion, bwv_label)
                                               + server_net.metric(logit_vs_fusion, vs_label), 7 * clinic_image.size(0))

            sps_acc_clic = torch.true_divide(server_net.metric(logit_pn_clic, pn_label)
                                             + server_net.metric(logit_str_clic, str_label)
                                             + server_net.metric(logit_pig_clic, pig_label)
                                             + server_net.metric(logit_rs_clic, rs_label)
                                             + server_net.metric(logit_dag_clic, dag_label)
                                             + server_net.metric(logit_bwv_clic, bwv_label)
                                             + server_net.metric(logit_vs_clic, vs_label), 7 * clinic_image.size(0))

            sps_acc_derm = torch.true_divide(server_net.metric(logit_pn_derm, pn_label)
                                             + server_net.metric(logit_str_derm, str_label)
                                             + server_net.metric(logit_pig_derm, pig_label)
                                             + server_net.metric(logit_rs_derm, rs_label)
                                             + server_net.metric(logit_dag_derm, dag_label)
                                             + server_net.metric(logit_bwv_derm, bwv_label)
                                             + server_net.metric(logit_vs_derm, vs_label), 7 * clinic_image.size(0))

            sps_acc = torch.true_divide(sps_acc_fusion + sps_acc_clic + sps_acc_derm, 3)

        val_loss += loss.item()
        val_dia_acc += dia_acc.item()
        val_sps_acc += sps_acc.item()

    num_batch = len(val_dataloader)
    val_loss = val_loss / num_batch
    val_dia_acc = val_dia_acc / num_batch
    val_sps_acc = val_sps_acc / num_batch
    return val_loss, val_dia_acc, val_sps_acc
