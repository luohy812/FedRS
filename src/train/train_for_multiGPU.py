from Loss.metric import eval
from utils.utils import resample_array, output2file, generate_gauss_weight

import torch.distributed as dist
from dataset.dataload_GPUS import *
from dataset.dataload import initialization
from train.train_method import train_consistency,train_consistency_socre
from init_setup.set_up import *
from Loss.our_loss import dice_and_ce_loss
from utils.vis_utils import loss_vis


import copy
def load_models(global_model, nodes, model_fname):
    global_model.load_state_dict(torch.load(model_fname)['global_model_state_dict'])
    for node_id in range(len(nodes)):
        nodes[node_id][0].load_state_dict(torch.load(model_fname)['local_model_{0:d}_state_dict'.format(node_id)])
        nodes[node_id][1].load_state_dict(torch.load(model_fname)['local_model_{0:d}_optimizer'.format(node_id)])
        nodes[node_id][2].load_state_dict(torch.load(model_fname)['local_model_{0:d}_scheduler'.format(node_id)])

def initialize_ood_node():
    ood_nodes = []
    for _, [node_name, d_name, d_path, fraction] in enumerate(cfg['ood_node_list']):
        folds, _ = create_folds(d_name, d_path, node_name, fraction, exclude_case=cfg['exclude_case'])

        # create out-of-distribution (ood) testing fold
        test_fold = folds[0]
        test_fold.extend(folds[1])
        test_fold.extend(folds[2])
        d_test = ClientDataset(test_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'],
                               rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'],
                               aug_data=False, full_sampling=True, enforce_fg=False, fixed_sample=False)
        dl_test = data.DataLoader(dataset=d_test, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True,
                                  drop_last=False, num_workers=cfg['cpu_thread'])

        print('{0:s}: test = {1:d}'.format(node_name, len(d_test)))

        ood_nodes.append([None, None, None, node_name, None, None, None, dl_test])

    return ood_nodes
def eval_global_model(val_loss_fn, val_result_fn, global_model, nodes, result_path, mode, commu_iters):
    t0 = time.perf_counter()

    gt_entries = []
    val_loss_list = []
    if mode == 'val':
        val_dice = 0.0
        val_dice_per_cls = np.zeros(cfg['cls_num'], dtype=float)
        val_num_per_cls = np.zeros(cfg['cls_num'], dtype=np.uint8)
        val_dice_num = 0
    global_model.eval()
    with torch.no_grad():
        for node_id, [local_model, optimizer, scheduler, node_name, node_weight, dl_train, dl_val, dl_test,
                      _] in enumerate(nodes):
            if mode == 'val':
                data_loader = dl_val
                metric_fname = 'metric_validation-{0:04d}'.format(commu_iters + 1)
                if torch.distributed.get_rank() == 0:
                    print(
                        'Validation ({0:d}/{1:d}) on Node #{2:d}'.format(commu_iters + 1, cfg['commu_times'], node_id))
            elif mode == 'test':
                data_loader = dl_test

                metric_fname = 'metric_testing'
                if torch.distributed.get_rank() == 0:
                    print('Testing on Node #{0:d}'.format(node_id))
            elif mode == 'test-ood':  # test on OoD (out-of-distribution) data
                data_loader = dl_test
                metric_fname = 'metric_testing_ood'

                print('Testing on OoD Node #{0:d}'.format(node_id))

            whole_prob_buffer = None
            whole_prob_weight = None
            batch_loss = 0
            batch_number = 0
            for batch_id, batch in enumerate(data_loader):
                image = batch['data']
                N = len(image)

                batch_number += 1

                image = image.cuda()
                # label = batch['label'].cuda()
                prob = global_model(image)
                # print(prob.dtype)
                # mask = torch.argmax(prob, dim=1, keepdim=True).detach().cpu().numpy().copy().astype(dtype=np.uint8)
                # mask = np.squeeze(mask, axis=1)
                if batch_id % 4 == 0:
                    print_line = '{0:s} --- Progress {1:5.2f}% (+{2:d})'.format(
                        mode, 100.0 * batch_id * cfg['Multi_test_batch_size'] / len(data_loader.dataset),
                              N * dist.get_world_size())
                    if torch.distributed.get_rank() == 0:
                        print(print_line)

                if mode == 'val':
                    label = batch['label'].cuda()
                    if torch.distributed.get_rank() == 0:
                        print(torch.unique(label))
                    loss_n = 0
                    for i in range(N):
                        d_name = batch['dataset'][i]

                        lmap = cfg['label_map'][d_name]
                        class_flag = [1]
                        for c in range(cfg['cls_num']):
                            if (c + 1) in lmap.values():
                                class_flag.append(1)
                            else:
                                class_flag.append(0)
                        # l_ce, l_dice, l_per_cls, n_per_cls = dice_and_ce_loss(prob[i:i + 1, :], label[i:i + 1, :],
                        #                                                       class_flag)
                        # loss_n += l_ce + l_dice
                        l_ce, l_dice, l_per_cls, n_per_cls = dice_and_ce_loss(prob[i:i + 1, :], label[i:i + 1, :],
                                                                              class_flag)

                        # batch_loss_list.append(batch_loss)
                        loss_n = l_dice + l_ce
                        val_dice += 1.0 - l_dice.item()
                        val_dice_per_cls += l_per_cls
                        val_num_per_cls += n_per_cls

                        del l_dice, l_per_cls, n_per_cls, l_ce
                    val_dice_num += N
                    batch_loss += loss_n
                    del loss_n
                    del label
                else:
                    # prob_tensor = prob.detach().cpu()
                    prob = prob.detach().cpu().numpy().copy()

                    for i in range(N):
                        # stack_size = torch.zeros_like(prob_tensor[i, :])
                        # stack_size = stack_size.numpy()
                        stack_size = batch['size'][i].numpy()
                        if torch.distributed.get_rank() == 0:
                            print(type(stack_size))
                            print(stack_size[0])
                            print(stack_size[1])
                            print(stack_size[2])
                            print(stack_size.shape)

                        stack_weight = generate_gauss_weight(stack_size[2], stack_size[1], stack_size[0])
                        if whole_prob_buffer is None:
                            whole_mask_size = stack_size.copy()
                            whole_mask_size[0] = (batch['patch_grid_size'][i][0] - 1) * batch['patch_stride'][i][0] + \
                                                 stack_size[0]
                            whole_mask_size[1] = (batch['patch_grid_size'][i][1] - 1) * batch['patch_stride'][i][1] + \
                                                 stack_size[1]
                            whole_mask_size[2] = (batch['patch_grid_size'][i][2] - 1) * batch['patch_stride'][i][2] + \
                                                 stack_size[2]
                            whole_mask_origin = batch['origin'][i].numpy()
                            whole_mask_spacing = batch['spacing'][i].numpy()
                            whole_prob_buffer = np.zeros(
                                (prob.shape[1], whole_mask_size[2], whole_mask_size[1], whole_mask_size[0]),
                                dtype=prob.dtype)
                            whole_prob_weight = np.zeros(
                                (prob.shape[1], whole_mask_size[2], whole_mask_size[1], whole_mask_size[0]),
                                dtype=prob.dtype)

                        stack_start_pos = [0, 0, 0]
                        stack_end_pos = [0, 0, 0]
                        stack_start_pos[0] = batch['patch_pos'][i][0] * batch['patch_stride'][i][0]
                        stack_start_pos[1] = batch['patch_pos'][i][1] * batch['patch_stride'][i][1]
                        stack_start_pos[2] = batch['patch_pos'][i][2] * batch['patch_stride'][i][2]
                        stack_end_pos[0] = stack_start_pos[0] + stack_size[0]
                        stack_end_pos[1] = stack_start_pos[1] + stack_size[1]
                        stack_end_pos[2] = stack_start_pos[2] + stack_size[2]
                        # if (whole_prob_buffer[:, stack_start_pos[2]:stack_end_pos[2],
                        #     stack_start_pos[1]:stack_end_pos[1], stack_start_pos[0]:stack_end_pos[0]]).shape != prob[i, :].shape:
                        #     whole_prob_buffer = np.zeros(prob[i,:])
                        if torch.distributed.get_rank() == 0:
                            print('prob', prob[i, :].shape)
                            print('whole', whole_prob_buffer[:, stack_start_pos[2]:stack_end_pos[2],
                                           stack_start_pos[1]:stack_end_pos[1],
                                           stack_start_pos[0]:stack_end_pos[0]].shape)
                        whole_prob_buffer[:, stack_start_pos[2]:stack_end_pos[2], stack_start_pos[1]:stack_end_pos[1],
                        stack_start_pos[0]:stack_end_pos[0]] += prob[i, :] * stack_weight

                        whole_prob_weight[:, stack_start_pos[2]:stack_end_pos[2], stack_start_pos[1]:stack_end_pos[1],
                        stack_start_pos[0]:stack_end_pos[0]] += stack_weight
                        if batch['eof'][i] == True:
                            whole_prob_buffer = whole_prob_buffer / whole_prob_weight
                            resampled_prob = np.zeros((whole_prob_buffer.shape[0], batch['org_size'][i][2],
                                                       batch['org_size'][i][1], batch['org_size'][i][0]),
                                                      dtype=prob.dtype)
                            for c in range(whole_prob_buffer.shape[0]):
                                resampled_prob[c, :] = resample_array(whole_prob_buffer[c, :], whole_mask_size,
                                                                      whole_mask_spacing, whole_mask_origin,
                                                                      batch['org_size'][i].numpy(),
                                                                      batch['org_spacing'][i].numpy(),
                                                                      batch['org_origin'][i].numpy(), linear=True)
                            whole_mask = resampled_prob.argmax(axis=0).astype(np.uint8)
                            print(np.unique(whole_mask))
                            output2file(whole_mask, batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(),
                                        batch['org_origin'][i].numpy(),
                                        '{}/{}@{}.nii.gz'.format(result_path, batch['dataset'][i], batch['case'][i]))
                            whole_prob_buffer = None
                            whole_prob_weight = None
                            gt_entries.append([batch['dataset'][i], batch['case'][i], batch['label_fname'][i]])

                del image, prob
            batch_loss = batch_loss / batch_number
            val_loss_list.append(batch_loss)
            # del batch_loss
        if mode == 'val':
            val_dice_per_cls = val_dice_per_cls / val_num_per_cls
            seg_dsc_m = val_dice_per_cls.mean()
            seg_dsc = None

            print_line = 'Validation result (iter = {0:d}/{1:d}) --- DSC {2:.2f}% ({3:s})'.format(
                commu_iters + 1, cfg['commu_times'], seg_dsc_m * 100.0,
                '/'.join(['%.2f'] * len(val_dice_per_cls)) % tuple(val_dice_per_cls * 100.0))
            print_line += '\n'
            loss_line = '{val_loss:s}'.format(val_loss='\t'.join(['%8.6f'] * len(val_loss_list)) % tuple(val_loss_list))
            loss_line += '\n'
            if torch.distributed.get_rank() == 0:
                with open(val_result_fn, 'a') as val_file:
                    val_file.write(print_line)
                with open(val_loss_fn, 'a') as val_loss_file:
                    val_loss_file.write(loss_line)

        else:
            seg_dsc, seg_asd, seg_hd, seg_dsc_m, seg_asd_m, seg_hd_m = eval(
                pd_path=result_path, gt_entries=gt_entries, label_map=cfg['label_map'], cls_num=cfg['cls_num'],
                metric_fn=metric_fname, calc_asd=(mode != 'val'))

            print_line = 'Testing results --- DSC {0:.2f} ({1:s})% --- ASD {2:.2f} ({3:s})mm --- HD {4:.2f} ({5:s})mm'.format(
                seg_dsc_m * 100.0, '/'.join(['%.2f'] * len(seg_dsc[:, 0])) % tuple(seg_dsc[:, 0] * 100.0),
                seg_asd_m, '/'.join(['%.2f'] * len(seg_asd[:, 0])) % tuple(seg_asd[:, 0]),
                seg_hd_m, '/'.join(['%.2f'] * len(seg_hd[:, 0])) % tuple(seg_hd[:, 0]))
            print_line += '\n'
            if torch.distributed.get_rank() == 0:
                with open(val_result_fn, 'a') as val_file:
                    val_file.write(print_line)

        t1 = time.perf_counter()
        eval_t = t1 - t0
    if torch.distributed.get_rank() == 0:
        print(print_line)
        print("Evaluation time cost: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
            h=int(eval_t) // 3600, m=(int(eval_t) % 3600) // 60, s=int(eval_t) % 60))

    return seg_dsc_m, seg_dsc
def communication(global_model, nodes, weight_list):
    # for each sub-encoder, we calculate the fusion weight according to the labeled client datasets
    with torch.no_grad():
        if torch.distributed.get_rank() == 0:
            print('score communicate')
        sub_encoder_weights = []

        for key in global_model.module.state_dict().keys():  # 模型参数 解决方案，让encoder不等于.sub_encoders
            temp = torch.zeros_like(global_model.module.state_dict()[key])  # 为当前键key创建一个与全局模型同形状的零张量temp。

            for node_id in range(len(nodes)):
                # print(label_size[node_id])
                temp += weight_list[node_id] * nodes[node_id][0].module.state_dict()[key]  # 4:由数据量大小分配的权重;0:参数

            global_model.module.state_dict()[key].data.copy_(temp)
            for node_id in range(len(nodes)):
                nodes[node_id][0].module.state_dict()[key].data.copy_(global_model.module.state_dict()[key])
    return global_model
def eval_global_model_test(val_result_fn, global_model, nodes, result_path, mode, commu_iters):
    t0 = time.perf_counter()
    global_model.eval()
    gt_entries = []
    if mode == 'val':
        val_dice = 0.0
        val_dice_per_cls = np.zeros(cfg['cls_num'], dtype=float)
        val_num_per_cls = np.zeros(cfg['cls_num'], dtype=np.uint8)
        val_dice_num = 0

    for node_id, [local_model, optimizer, scheduler, node_name, node_weight, dl_train, dl_val, dl_test] in enumerate(
            nodes):
        if mode == 'val':
            data_loader = dl_val
            metric_fname = 'metric_validation-{0:04d}'.format(commu_iters + 1)

            print('Validation ({0:d}/{1:d}) on Node #{2:d}'.format(commu_iters + 1, cfg['commu_times'], node_id))
        elif mode == 'test':
            data_loader = dl_test
            metric_fname = 'metric_testing'

            print('Testing on Node #{0:d}'.format(node_id))
        elif mode == 'test-ood':  # test on OoD (out-of-distribution) data
            data_loader = dl_test
            metric_fname = 'metric_testing_ood'

            print('Testing on OoD Node #{0:d}'.format(node_id))

        whole_prob_buffer = None
        whole_prob_weight = None

        for batch_id, batch in enumerate(data_loader):
                image = batch['data']
                N = len(image)

                image = image.cuda()
                # label = batch['label'].cuda()
                # print(torch.unique(label))

                prob = global_model(image)

                # mask = torch.argmax(prob, dim=1, keepdim=True).detach().cpu().numpy().copy().astype(dtype=np.uint8)
                # mask = np.squeeze(mask, axis=1)
                rank = dist.get_rank()

                print_line = f'Rank {rank}:'+'\t'
                print_line += '{0:s} --- Progress {1:5.2f}% (+{2:d})'.format(
                    mode, 100.0 * batch_id * cfg['test_batch_size'] / len(data_loader.dataset), N)
                print(print_line)

                if mode == 'val':
                    label = batch['label'].cuda()

                    for i in range(N):
                        d_name = batch['dataset'][i]

                        lmap = cfg['label_map'][d_name]
                        class_flag = [1]
                        for c in range(cfg['cls_num']):
                            if (c + 1) in lmap.values():
                                class_flag.append(1)
                            else:
                                class_flag.append(0)
                        _, l_dice, l_per_cls, n_per_cls = dice_and_ce_loss(prob[i:i + 1, :], label[i:i + 1, :], class_flag)
                        val_dice += 1.0 - l_dice.item()
                        val_dice_per_cls += l_per_cls
                        val_num_per_cls += n_per_cls
                        del l_dice, l_per_cls, n_per_cls
                    val_dice_num += N
                    del label
                else:

                    prob = prob.detach().cpu().numpy().copy()

                    for i in range(N):
                        stack_size = batch['size'][i].numpy()

                        stack_weight = generate_gauss_weight(stack_size[2], stack_size[1], stack_size[0])
                        if whole_prob_buffer is None:
                            whole_mask_size = stack_size.copy()
                            # print(stack_size[0])
                            # print(batch['patch_grid_size'][i][0] - 1)
                            # print(batch['patch_grid_size'][i][0])
                            whole_mask_size[0] = (batch['patch_grid_size'][i][0] - 1) * batch['patch_stride'][i][0] + \
                                                 stack_size[0]
                            whole_mask_size[1] = (batch['patch_grid_size'][i][1] - 1) * batch['patch_stride'][i][1] + \
                                                 stack_size[1]
                            whole_mask_size[2] = (batch['patch_grid_size'][i][2] - 1) * batch['patch_stride'][i][2] + \
                                                 stack_size[2]
                            whole_mask_origin = batch['origin'][i].numpy()
                            whole_mask_spacing = batch['spacing'][i].numpy()
                            whole_prob_buffer = np.zeros(
                                (prob.shape[1], whole_mask_size[2], whole_mask_size[1], whole_mask_size[0]),
                                dtype=prob.dtype)
                            whole_prob_weight = np.zeros(
                                (prob.shape[1], whole_mask_size[2], whole_mask_size[1], whole_mask_size[0]),
                                dtype=prob.dtype)

                        stack_start_pos = [0, 0, 0]
                        stack_end_pos = [0, 0, 0]
                        stack_start_pos[0] = batch['patch_pos'][i][0] * batch['patch_stride'][i][0]
                        stack_start_pos[1] = batch['patch_pos'][i][1] * batch['patch_stride'][i][1]
                        stack_start_pos[2] = batch['patch_pos'][i][2] * batch['patch_stride'][i][2]
                        stack_end_pos[0] = stack_start_pos[0] + stack_size[0]
                        stack_end_pos[1] = stack_start_pos[1] + stack_size[1]
                        stack_end_pos[2] = stack_start_pos[2] + stack_size[2]
                        # if (whole_prob_buffer[:, stack_start_pos[2]:stack_end_pos[2],
                        #     stack_start_pos[1]:stack_end_pos[1], stack_start_pos[0]:stack_end_pos[0]]).shape == prob[i, :].shape:
                        whole_prob_buffer[:, stack_start_pos[2]:stack_end_pos[2], stack_start_pos[1]:stack_end_pos[1],
                        stack_start_pos[0]:stack_end_pos[0]] += prob[i, :] * stack_weight

                        whole_prob_weight[:, stack_start_pos[2]:stack_end_pos[2], stack_start_pos[1]:stack_end_pos[1],
                        stack_start_pos[0]:stack_end_pos[0]] += stack_weight
                        if batch['eof'][i] == True:
                            whole_prob_buffer = whole_prob_buffer / whole_prob_weight
                            resampled_prob = np.zeros((whole_prob_buffer.shape[0], batch['org_size'][i][2],
                                                       batch['org_size'][i][1], batch['org_size'][i][0]), dtype=prob.dtype)
                            for c in range(whole_prob_buffer.shape[0]):
                                resampled_prob[c, :] = resample_array(whole_prob_buffer[c, :], whole_mask_size,
                                                                      whole_mask_spacing, whole_mask_origin,
                                                                      batch['org_size'][i].numpy(),
                                                                      batch['org_spacing'][i].numpy(),
                                                                      batch['org_origin'][i].numpy(), linear=True)
                            whole_mask = resampled_prob.argmax(axis=0).astype(np.uint8)
                            output2file(whole_mask, batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(),
                                        batch['org_origin'][i].numpy(),
                                        '{}/{}@{}.nii.gz'.format(result_path, batch['dataset'][i], batch['case'][i]))
                            whole_prob_buffer = None
                            whole_prob_weight = None
                            gt_entries.append([batch['dataset'][i], batch['case'][i], batch['label_fname'][i]])

        del image, prob

    if mode == 'val':
        val_dice_per_cls = val_dice_per_cls / val_num_per_cls
        seg_dsc_m = val_dice_per_cls.mean()
        seg_dsc = None

        print_line = 'Validation result (iter = {0:d}/{1:d}) --- DSC {2:.2f}% ({3:s})'.format(
            commu_iters + 1, cfg['commu_times'], seg_dsc_m * 100.0,
            '/'.join(['%.2f'] * len(val_dice_per_cls)) % tuple(val_dice_per_cls * 100.0))
        print_line += '\n'
        with open(val_result_fn, 'a') as val_file:
            val_file.write(print_line)

    else:
        seg_dsc, seg_asd, seg_hd, seg_dsc_m, seg_asd_m, seg_hd_m = eval(
            pd_path=result_path, gt_entries=gt_entries, label_map=cfg['label_map'], cls_num=cfg['cls_num'],
            metric_fn=metric_fname, calc_asd=(mode != 'val'))
        rank = dist.get_rank()
        print_line = f'Rank {rank}:' + '\t'
        print_line += 'Testing results --- DSC {0:.2f} ({1:s})% --- ASD {2:.2f} ({3:s})mm --- HD {4:.2f} ({5:s})mm'.format(
            seg_dsc_m * 100.0, '/'.join(['%.2f'] * len(seg_dsc[:, 0])) % tuple(seg_dsc[:, 0] * 100.0),
            seg_asd_m, '/'.join(['%.2f'] * len(seg_asd[:, 0])) % tuple(seg_asd[:, 0]),
            seg_hd_m, '/'.join(['%.2f'] * len(seg_hd[:, 0])) % tuple(seg_hd[:, 0]))
        print_line += '\n'
        with open(val_result_fn, 'a') as val_file:
            val_file.write(print_line)
    print(print_line)
    t1 = time.perf_counter()
    eval_t = t1 - t0
    print("Evaluation time cost: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
        h=int(eval_t) // 3600, m=(int(eval_t) % 3600) // 60, s=int(eval_t) % 60))

    return seg_dsc_m, seg_dsc
def train_new_GPUs_ours():
    args = setup_multiGPU()
    print_args = str(args)
    seed = cfg['seed']
    rank = torch.distributed.get_rank()
    set_seed(seed + rank)
    store_dir, train_start_time, val_result_fn, \
        val_result_path, loss_fn, time_stamp, test_result_path, val_loss_fn = set_dir()
    weight_fn = '{}/weight.txt'.format(store_dir)
    threshold_fn = '{}/threshold.txt'.format(store_dir)
    global_model, nodes, node_cls_flag, mode = initialization_gpus(args)
    ma_val_acc = None
    best_val_acc = 0
    start_iter = 0
    acc_time = 0
    best_model_fn = '{0:s}/cp_commu_{1:04d}.pth.tar'.format(store_dir, 1)
    final_model_fn = '{0:s}/cp_commu_{1:04d}.pth.tar'.format(store_dir, 1)
    log_line = "Model: {}\nModel parameters: {}\nStart time: {}\nConfiguration:\n".format(
        global_model.module.description(),
        sum(x.numel() for x in global_model.parameters()),
        time.strftime("%Y-%m-%d %H:%M:%S", train_start_time))

    for cfg_key in cfg:
        log_line += ' --- {}: {}\n'.format(cfg_key, cfg[cfg_key])
    if torch.distributed.get_rank() == 0:
        print(log_line)
        log_line += '\n'

        with open(val_result_fn, 'a') as val_file:
            val_file.write(log_line)
            val_file.write(print_args + '\n')

    fed_avg = True
    for commu_t in range(start_iter, cfg['commu_times'], 1):
        consistency_list = torch.zeros(cfg['cls_num']).float()
        consistency_list = consistency_list.cuda()
        photo_number_list = torch.zeros(cfg['cls_num']).float()
        photo_number_list = photo_number_list.cuda()
        t0 = time.perf_counter()
        train_loss = []
        for node_id, [local_model, optimizer, scheduler, node_name, node_weight, dl_train, dl_val,
                      dl_test, train_sampler] in enumerate(nodes):
            if torch.distributed.get_rank() == 0:
                print('Training ({0:d}/{1:d}) on Node: {2:s}\n'.format(commu_t + 1, cfg['commu_times'], node_name))

            local_model.train()
            if torch.distributed.get_rank() == 0:
                print("ours train")
            train_loss.append(
                train_consistency(train_sampler, local_model, global_model, optimizer,
                                  scheduler, dl_train, cfg['epoch_per_commu'], commu_t))
            train_consistency_socre(train_sampler, local_model, global_model, dl_train, cfg['epoch_per_commu'],
                                    consistency_list,photo_number_list,commu_t)

        consistency_list = consistency_list / photo_number_list
        consistency_list = consistency_list
        total_score = sum(consistency_list)
        consistency_list = [score / total_score for score in consistency_list]
        if torch.distributed.get_rank() == 0:
            values = torch.tensor([t.item() for t in consistency_list])
            with open(weight_fn, 'a') as val_file:
                val_file.write(str(values) + '\n')

        communication(global_model, nodes, consistency_list)

        t1 = time.perf_counter()
        epoch_t = t1 - t0
        acc_time += epoch_t
        if torch.distributed.get_rank() == 0:
            print("Iteration time cost: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
                h=int(epoch_t) // 3600, m=(int(epoch_t) % 3600) // 60, s=int(epoch_t) % 60))

        if commu_t % 1 == 0:
            seg_dsc_m, _ = eval_global_model(val_loss_fn, val_result_fn, global_model, nodes, val_result_path,
                                             mode='val',
                                             commu_iters=commu_t)
            if ma_val_acc is None:
                ma_val_acc = seg_dsc_m
            else:
                ma_val_acc = seg_dsc_m

            loss_line = '{commu_iter:>04d}\t{train_loss:s}\t{seg_val_dsc:>8.6f}\t{ma_val_dsc:>8.6f}'.format(
                commu_iter=commu_t + 1, train_loss='\t'.join(['%8.6f'] * len(train_loss)) % tuple(train_loss),
                seg_val_dsc=seg_dsc_m, ma_val_dsc=ma_val_acc)
            for node_id, [_, _, scheduler, _, _, _, _, _, _] in enumerate(nodes):
                loss_line += '\t{node_lr:>8.6f}'.format(node_lr=scheduler.get_last_lr()[0])
            loss_line += '\n'
            if torch.distributed.get_rank() == 0:
                with open(loss_fn, 'a') as loss_file:
                    loss_file.write(loss_line)
                # show train_loss:
                loss_vis(store_dir, loss_fn, 'train', commu_t)
                loss_vis(store_dir, val_loss_fn, 'val', commu_t)

            if torch.distributed.get_rank() == 0:
                # save best model
                if commu_t == 0 or ma_val_acc > best_val_acc:
                    # remove former best model
                    if os.path.exists(best_model_fn):
                        os.remove(best_model_fn)
                    # save current best model
                    best_val_acc = ma_val_acc
                    best_model_fn = '{0:s}/best_cp_commu_{1:04d}.pth.tar'.format(store_dir, commu_t + 1)
                    best_model_cp = {
                        'commu_iter': commu_t,
                        'acc_time': acc_time,
                        'time_stamp': time_stamp,
                        'best_val_acc': best_val_acc,
                        'best_model_filename': best_model_fn,
                        'global_model_state_dict': global_model.state_dict()}
                    for node_id, [local_model, optimizer, scheduler, _, _, _, _, _, _] in enumerate(nodes):
                        best_model_cp['local_model_{0:d}_state_dict'.format(node_id)] = local_model.state_dict()
                        best_model_cp['local_model_{0:d}_optimizer'.format(node_id)] = optimizer.state_dict()
                        best_model_cp['local_model_{0:d}_scheduler'.format(node_id)] = scheduler.state_dict()
                    torch.save(best_model_cp, best_model_fn)
                    if torch.distributed.get_rank() == 0:
                        print('Best model (communication iteration = {}) saved.\n'.format(commu_t + 1))
            if torch.distributed.get_rank() == 2:
                if commu_t == cfg['commu_times'] - 1:
                    final_val_acc = ma_val_acc
                    final_model_fn = '{0:s}/final_cp_commu_{1:04d}.pth.tar'.format(store_dir, commu_t + 1)
                    final_model_cp = {
                        'commu_iter': commu_t,
                        'acc_time': acc_time,
                        'time_stamp': time_stamp,
                        'best_val_acc': final_val_acc,
                        'best_model_filename': final_model_fn,
                        'global_model_state_dict': global_model.state_dict()}
                    for node_id, [local_model, optimizer, scheduler, _, _, _, _, _, _] in enumerate(nodes):
                        final_model_cp['local_model_{0:d}_state_dict'.format(node_id)] = local_model.state_dict()
                        final_model_cp['local_model_{0:d}_optimizer'.format(node_id)] = optimizer.state_dict()
                        final_model_cp['local_model_{0:d}_scheduler'.format(node_id)] = scheduler.state_dict()
                    torch.save(final_model_cp, final_model_fn)
                    if torch.distributed.get_rank() == 0:
                        print('Final model (communication iteration = {}) saved.\n'.format(commu_t + 1))

        else:
            loss_line = '{commu_iter:>04d}\t{train_loss:s}'.format(
                commu_iter=commu_t + 1, train_loss='\t'.join(['%8.6f'] * len(train_loss)) % tuple(train_loss))
            for node_id, [_, _, scheduler, _, _, _, _, _, _] in enumerate(nodes):
                loss_line += '\t{node_lr:>8.6f}'.format(node_lr=scheduler.get_last_lr()[0])
            loss_line += '\n'
            with open(loss_fn, 'a') as loss_file:
                loss_file.write(loss_line)

    print_line = "Total training time: {h:>02d}:{m:>02d}:{s:>02d}\n\n".format(
        h=int(acc_time) // 3600, m=(int(acc_time) % 3600) // 60, s=int(acc_time) % 60)
    if torch.distributed.get_rank() == 0:
        print(print_line)

        with open(val_result_fn, 'a') as val_file:
            val_file.write(print_line)

    best_test_result_path = '{}/results_test/best'.format(store_dir)
    os.makedirs(best_test_result_path, exist_ok=True)
    final_test_result_path = '{}/results_test/final'.format(store_dir)
    os.makedirs(final_test_result_path, exist_ok=True)
    ood_nodes = initialize_ood_node()
    if torch.distributed.get_rank() == 0:
        global_model, nodes, node_cls_flag = initialization(gpu=[0, 1])
        print('best', str(best_model_fn))
        with open(val_result_fn, 'a') as val_file:
            val_file.write(str(best_model_fn) + '\n')
        load_models(global_model, nodes, best_model_fn)
        eval_global_model_test(val_result_fn, global_model, nodes, best_test_result_path, mode='test',
                               commu_iters=0)
        eval_global_model_test(val_result_fn, global_model, ood_nodes, best_test_result_path, mode='test-ood',
                               commu_iters=0)
        best_in_txt = '{}/metric_testing.txt'.format(best_test_result_path)
        best_out_txt = '{}/metric_testing_ood.txt'.format(best_test_result_path)


    dist.destroy_process_group()








