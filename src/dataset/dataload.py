from dataset.dataset import create_folds, ClientDataset
from torch.utils import data
from init_setup.initialization import cfg
import torch.nn as nn

from models.normal_UNet import Unet
from torch import optim

def initial_net(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialization(gpu):
    global_model = nn.DataParallel(module=Unet(in_ch=1, base_ch=cfg['base_ch'], cls_num=cfg['cls_num']), device_ids=gpu)
    global_model.cuda()
    initial_net(global_model)

    nodes = []
    weight_sum = 0
    node_cls_flag = []
    for node_id, [node_name, d_name, d_path, fraction] in enumerate(cfg['node_list']):

        folds, _ = create_folds(d_name, d_path, node_name, fraction, exclude_case=cfg['exclude_case'])

        # create training fold
        train_fold = folds[0]
        d_train = ClientDataset(train_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'],
                                rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'],
                                aug_data=True, full_sampling=False, enforce_fg=False, fixed_sample=False)

        dl_train = data.DataLoader(dataset=d_train, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True,
                                   drop_last=True, num_workers=cfg['cpu_thread'])

        # create validaion fold
        val_fold = folds[1]
        d_val = ClientDataset(val_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'],
                              rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'],
                              aug_data=False, full_sampling=False, enforce_fg=True, fixed_sample=True)

        dl_val = data.DataLoader(dataset=d_val, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True,
                                 drop_last=False, num_workers=cfg['cpu_thread'])

        # create testing fold
        test_fold = folds[2]
        d_test = ClientDataset(test_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'],
                               rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'],
                               aug_data=False, full_sampling=True, enforce_fg=False, fixed_sample=False)

        dl_test = data.DataLoader(dataset=d_test, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True,
                                  drop_last=False, num_workers=cfg['cpu_thread'])

        print('{0:s}: train/val/test = {1:d}/{2:d}/{3:d}'.format(node_name, len(d_train), len(d_val), len(d_test)))
        weight_sum += len(d_train)

        local_model = nn.DataParallel(module=Unet(in_ch=1, base_ch=cfg['base_ch'], cls_num=cfg['cls_num']),device_ids=gpu)
        local_model.cuda()
        local_model.load_state_dict(global_model.state_dict())


        node_enabled_encoders = []
        for c in range(cfg['cls_num']):
            if (c + 1) in cfg['label_map'][d_name[0]].values():
                node_enabled_encoders.append(c)

        optimizer1 = optim.SGD(local_model.module.get_s1_parameters(), lr=cfg['lr'], momentum=0.99,
                               nesterov=True, weight_decay=cfg['weight_decay'])
        # optimizer2 = optim.SGD(local_model.module.get_s2_parameters(node_enabled_encoders), lr=cfg['lr'], momentum=0.99,
        #                        nesterov=True, weight_decay=3e-5)

        lambda_func = lambda epoch: (1 - epoch / (cfg['commu_times'] * cfg['epoch_per_commu'])) ** 0.9
        scheduler1 = optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=lambda_func)
        # scheduler2 = optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda=lambda_func)

        nodes.append(
            [local_model, optimizer1, scheduler1, node_name, len(d_train), dl_train, dl_val,
             dl_test])

        cls_labels = []
        for dn in d_name:
            cls_labels.extend(cfg['label_map'][dn].values())
        tmp_flag = []
        for c in range(cfg['cls_num']):
            if (c + 1) in cls_labels:
                tmp_flag.append(True)
            else:
                tmp_flag.append(False)
        node_cls_flag.append(tmp_flag)

    for i in range(len(nodes)):
        nodes[i][4] = nodes[i][4] / weight_sum
        print('Weight of {0:s}: {1:f}'.format(nodes[i][3], nodes[i][4]))

    return global_model, nodes, node_cls_flag
