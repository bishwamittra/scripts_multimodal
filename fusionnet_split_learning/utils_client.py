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

            dia_fusion_acc = torch.true_divide(server_net.metric(
                logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))
            dia_clic_acc = torch.true_divide(server_net.metric(
                logit_diagnosis_clic, diagnosis_label), clinic_image.size(0))
            dia_derm_acc = torch.true_divide(server_net.metric(
                logit_diagnosis_derm, diagnosis_label), clinic_image.size(0))

            dia_acc = torch.true_divide(
                dia_fusion_acc + dia_clic_acc + dia_derm_acc, 3)

            sps_fusion_acc = torch.true_divide(server_net.metric(logit_pn_fusion, pn_label)
                                               + server_net.metric(logit_str_fusion, str_label)
                                               + server_net.metric(logit_pig_fusion, pig_label)
                                               + server_net.metric(logit_rs_fusion, rs_label)
                                               + server_net.metric(logit_dag_fusion, dag_label)
                                               + server_net.metric(logit_bwv_fusion, bwv_label)
                                               + server_net.metric(logit_vs_fusion, vs_label), 7 * clinic_image.size(0))

            sps_clic_acc = torch.true_divide(server_net.metric(logit_pn_clic, pn_label)
                                             + server_net.metric(logit_str_clic, str_label)
                                             + server_net.metric(logit_pig_clic, pig_label)
                                             + server_net.metric(logit_rs_clic, rs_label)
                                             + server_net.metric(logit_dag_clic, dag_label)
                                             + server_net.metric(logit_bwv_clic, bwv_label)
                                             + server_net.metric(logit_vs_clic, vs_label), 7 * clinic_image.size(0))

            sps_derm_acc = torch.true_divide(server_net.metric(logit_pn_derm, pn_label)
                                             + server_net.metric(logit_str_derm, str_label)
                                             + server_net.metric(logit_pig_derm, pig_label)
                                             + server_net.metric(logit_rs_derm, rs_label)
                                             + server_net.metric(logit_dag_derm, dag_label)
                                             + server_net.metric(logit_bwv_derm, bwv_label)
                                             + server_net.metric(logit_vs_derm, vs_label), 7 * clinic_image.size(0))

            sps_acc = torch.true_divide(sps_fusion_acc + sps_clic_acc + sps_derm_acc, 3)

            # break

        val_loss += loss.item()
        val_dia_acc += dia_acc.item()
        val_sps_acc += sps_acc.item()

    num_batch = len(val_dataloader)
    val_loss = val_loss / num_batch
    val_dia_acc = val_dia_acc / num_batch
    val_sps_acc = val_sps_acc / num_batch
    return val_loss, val_dia_acc, val_sps_acc



def validation_u_shaped(client_first_model, client_last_model, server_middle_model, val_dataloader, device):
    client_first_model.set_mode('valid')
    server_middle_model.set_mode('valid')
    client_last_model.set_mode('valid')
    
    val_loss = 0
    val_dia_acc = 0
    val_sps_acc = 0
    val_pn_acc = 0
    val_str_acc = 0
    val_pig_acc = 0
    val_rs_acc = 0
    val_dag_acc = 0
    val_bwv_acc = 0
    val_vs_acc = 0

    val_dia_clic_acc = 0
    val_sps_clic_acc = 0
    val_pn_clic_acc = 0
    val_str_clic_acc = 0
    val_pig_clic_acc = 0
    val_rs_clic_acc = 0
    val_dag_clic_acc = 0
    val_bwv_clic_acc = 0
    val_vs_clic_acc = 0

    val_dia_derm_acc = 0
    val_sps_derm_acc = 0
    val_pn_derm_acc = 0
    val_str_derm_acc = 0
    val_pig_derm_acc = 0
    val_rs_derm_acc = 0
    val_dag_derm_acc = 0
    val_bwv_derm_acc = 0
    val_vs_derm_acc = 0

    val_dia_fusion_acc = 0
    val_sps_fusion_acc = 0
    val_pn_fusion_acc = 0
    val_str_fusion_acc = 0
    val_pig_fusion_acc = 0
    val_rs_fusion_acc = 0
    val_dag_fusion_acc = 0
    val_bwv_fusion_acc = 0
    val_vs_fusion_acc = 0

    for (clinic_image, derm_image, meta_data, label) in tqdm(val_dataloader):

        clinic_image = clinic_image.to(device)
        derm_image = derm_image.to(device)
        meta_data = meta_data.to(device)

        with torch.no_grad():

            
            client_first_output = client_first_model((clinic_image, derm_image))
            server_middle_output = server_middle_model(client_first_output)
            loss, \
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
                [vs_clic_acc, vs_derm_acc, vs_fusion_acc]
            ] = client_last_model.forward_propagate_and_loss_compute(server_middle_output, label, clinic_image.size(0), device)
            
            val_loss += loss.item()
            val_dia_acc += dia_acc.item()
            val_sps_acc += sps_acc.item()
            val_pn_acc += pn_acc.item()
            val_str_acc += str_acc.item()
            val_pig_acc += pig_acc.item()
            val_rs_acc += rs_acc.item()
            val_dag_acc += dag_acc.item()
            val_bwv_acc += bwv_acc.item()
            val_vs_acc += vs_acc.item()

            val_dia_clic_acc += dia_clic_acc.item()
            val_sps_clic_acc += sps_clic_acc.item()
            val_pn_clic_acc += pn_clic_acc.item()
            val_str_clic_acc += str_clic_acc.item()
            val_pig_clic_acc += pig_clic_acc.item()
            val_rs_clic_acc += rs_clic_acc.item()
            val_dag_clic_acc += dag_clic_acc.item()
            val_bwv_clic_acc += bwv_clic_acc.item()
            val_vs_clic_acc += vs_clic_acc.item()

            val_dia_derm_acc += dia_derm_acc.item()
            val_sps_derm_acc += sps_derm_acc.item()
            val_pn_derm_acc += pn_derm_acc.item()
            val_str_derm_acc += str_derm_acc.item()
            val_pig_derm_acc += pig_derm_acc.item()
            val_rs_derm_acc += rs_derm_acc.item()
            val_dag_derm_acc += dag_derm_acc.item()
            val_bwv_derm_acc += bwv_derm_acc.item()
            val_vs_derm_acc += vs_derm_acc.item()
            
            val_dia_fusion_acc += dia_fusion_acc.item()
            val_sps_fusion_acc += sps_fusion_acc.item()
            val_pn_fusion_acc += pn_fusion_acc.item()
            val_str_fusion_acc += str_fusion_acc.item()
            val_pig_fusion_acc += pig_fusion_acc.item()
            val_rs_fusion_acc += rs_fusion_acc.item()
            val_dag_fusion_acc += dag_fusion_acc.item()
            val_bwv_fusion_acc += bwv_fusion_acc.item()
            val_vs_fusion_acc += vs_fusion_acc.item()


            

    num_batch = len(val_dataloader)
    val_loss = val_loss / num_batch
    val_dia_acc = val_dia_acc / num_batch
    val_sps_acc = val_sps_acc / num_batch
    val_pn_acc = val_pn_acc / num_batch
    val_str_acc = val_str_acc / num_batch
    val_pig_acc = val_pig_acc / num_batch
    val_rs_acc = val_rs_acc / num_batch
    val_dag_acc = val_dag_acc / num_batch
    val_bwv_acc = val_bwv_acc / num_batch
    val_vs_acc = val_vs_acc / num_batch

    val_dia_clic_acc = val_dia_clic_acc / num_batch
    val_sps_clic_acc = val_sps_clic_acc / num_batch
    val_pn_clic_acc = val_pn_clic_acc / num_batch
    val_str_clic_acc = val_str_clic_acc / num_batch
    val_pig_clic_acc = val_pig_clic_acc / num_batch
    val_rs_clic_acc = val_rs_clic_acc / num_batch
    val_dag_clic_acc = val_dag_clic_acc / num_batch
    val_bwv_clic_acc = val_bwv_clic_acc / num_batch
    val_vs_clic_acc = val_vs_clic_acc / num_batch

    val_dia_derm_acc = val_dia_derm_acc / num_batch
    val_sps_derm_acc = val_sps_derm_acc / num_batch
    val_pn_derm_acc = val_pn_derm_acc / num_batch
    val_str_derm_acc = val_str_derm_acc / num_batch
    val_pig_derm_acc = val_pig_derm_acc / num_batch
    val_rs_derm_acc = val_rs_derm_acc / num_batch
    val_dag_derm_acc = val_dag_derm_acc / num_batch
    val_bwv_derm_acc = val_bwv_derm_acc / num_batch
    val_vs_derm_acc = val_vs_derm_acc / num_batch

    val_dia_fusion_acc = val_dia_fusion_acc / num_batch
    val_sps_fusion_acc = val_sps_fusion_acc / num_batch
    val_pn_fusion_acc = val_pn_fusion_acc / num_batch
    val_str_fusion_acc = val_str_fusion_acc / num_batch
    val_pig_fusion_acc = val_pig_fusion_acc / num_batch
    val_rs_fusion_acc = val_rs_fusion_acc / num_batch
    val_dag_fusion_acc = val_dag_fusion_acc / num_batch
    val_bwv_fusion_acc = val_bwv_fusion_acc / num_batch
    val_vs_fusion_acc = val_vs_fusion_acc / num_batch

    # return val_loss, val_dia_acc, val_sps_acc

    return val_loss, \
        val_dia_acc, \
        [val_dia_clic_acc, val_dia_derm_acc, val_dia_fusion_acc], \
        val_sps_acc, \
        [val_sps_clic_acc, val_sps_derm_acc, val_sps_fusion_acc], \
        [val_pn_acc, val_str_acc, val_pig_acc, val_rs_acc, val_dag_acc, val_bwv_acc, val_vs_acc], \
        [[val_pn_clic_acc, val_pn_derm_acc, val_pn_fusion_acc], 
            [val_str_clic_acc, val_str_derm_acc, val_str_fusion_acc], 
            [val_pig_clic_acc, val_pig_derm_acc, val_pig_fusion_acc], 
            [val_rs_clic_acc, val_rs_derm_acc, val_rs_fusion_acc], 
            [val_dag_clic_acc, val_dag_derm_acc, val_dag_fusion_acc], 
            [val_bwv_clic_acc, val_bwv_derm_acc, val_bwv_fusion_acc], 
            [val_vs_clic_acc, val_vs_derm_acc, val_vs_fusion_acc]]
