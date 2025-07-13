# 文件名: utils/multiGAN_trainer_disccls.py

import torch.nn as nn
import copy
from .evaluate_visualization import *
import torch.optim.lr_scheduler as lr_scheduler
import time
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from .util import get_autocast_context
import logging
from torch.cuda.amp import GradScaler
import numpy as np
import models  # 确保导入 models 模块


# scaler = GradScaler()  <--- 已从此处移除


def get_loss_function(name):
    if name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif name == 'mse':
        return nn.MSELoss()
    elif name == 'mae':
        return nn.L1Loss()
    elif name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"未知的损失函数名称: {name}")


def train_multi_gan(args, generators, discriminators, dataloaders, window_sizes, y_scaler, train_xes, train_y, val_xes,
                    val_y, val_y_gan, val_label_gan, output_dir, device,
                    init_GDweight=[[1, 0, 0, 1.0], [0, 1, 0, 1.0], [0, 0, 1, 1.0]],
                    final_GDweight=[[0.333, 0.333, 0.333, 1.0], [0.333, 0.333, 0.333, 1.0], [0.333, 0.333, 0.333, 1.0]],
                    logger=None, dynamic_weight=False, date_series=None):
    # 解决方案: 在函数开始时创建 GradScaler 实例，确保每次调用都是全新的、独立的
    scaler = GradScaler()

    N = len(generators)
    num_epochs = args.num_epochs
    distill_epochs = args.distill_epochs
    cross_finetune_epochs = args.cross_finetune_epochs

    adv_criterion, reg_criterion, cls_criterion = get_loss_function(args.adversarial_loss_mode), get_loss_function(
        args.regression_loss_mode), get_loss_function('cross_entropy')

    optimizers_G = []
    for gen in generators:
        if isinstance(gen, models.Generator_mpd):
            finetune_lr = args.lr * args.lr_cae_finetune_multiplier
            optimizer = torch.optim.AdamW([
                {'params': gen.cae_encoder.parameters(), 'lr': finetune_lr},
                {'params': gen.to_transformer_input.parameters(), 'lr': args.lr},
                {'params': gen.transformer_encoder.parameters(), 'lr': args.lr},
                {'params': gen.regression_head.parameters(), 'lr': args.lr},
                {'params': gen.classification_head.parameters(), 'lr': args.lr}
            ], betas=(0.9, 0.999))
            optimizers_G.append(optimizer)
            print(f"为 Generator_mpd 设置了分层学习率: CAE部分学习率为 {finetune_lr}, 其他部分为 {args.lr}")
        else:
            optimizers_G.append(torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.9, 0.999)))

    monitor_mode = 'max' if args.monitor_metric == 'val_acc' else 'min'
    schedulers = [lr_scheduler.ReduceLROnPlateau(opt, mode=monitor_mode, factor=0.1, patience=16, min_lr=1e-7) for opt
                  in optimizers_G]
    optimizers_D = [torch.optim.Adam(m.parameters(), lr=args.lr, betas=(0.9, 0.999)) for m in discriminators]

    best_epoch = [-1] * N
    keys = [];
    g_keys, d_keys, MSE_g_keys, val_loss_keys, acc_keys = [f'G{i}' for i in range(1, N + 1)], [f'D{i}' for i in
                                                                                               range(1, N + 1)], [
        f'MSE_G{i}' for i in range(1, N + 1)], [f'val_G{i}' for i in range(1, N + 1)], [f'acc_G{i}' for i in
                                                                                        range(1, N + 1)]
    keys.extend(g_keys);
    keys.extend(d_keys);
    keys.extend(MSE_g_keys);
    keys.extend(val_loss_keys);
    keys.extend(acc_keys)
    d_g_keys = [f'D{i}_G{j}' for i in range(1, N + 1) for j in range(1, N + 1)];
    keys.extend(d_g_keys)
    hists_dict = {k: np.zeros(num_epochs) for k in keys}
    best_monitor_metric = [float('-inf') if args.monitor_metric == 'val_acc' else float('inf') for _ in range(N)]
    best_model_state, patience_counter, patience = [None] * N, 0, 15
    print("开始训练")

    is_mpd_run = "mpd" in args.generators[0]

    for epoch in range(num_epochs):
        epo_start = time.time()
        weight_matrix = torch.tensor(init_GDweight if epoch < 10 else final_GDweight).to(device)
        loss_dict = {k: [] for k in keys if k not in val_loss_keys and k not in acc_keys}
        gaps = [window_sizes[-1] - w for w in window_sizes]

        for batch_idx, (x_batch_data, y_last, label_last) in enumerate(dataloaders[-1]):
            if label_last.dim() == 2:
                label_last = label_last.unsqueeze(-1)

            x_batch_data, y_last, label_last = x_batch_data.to(device), y_last.to(device), label_last.to(device)

            X, Y, LABELS = [], [], []

            if is_mpd_run:
                for gap in gaps:
                    X.append(x_batch_data[:, :, gap:, :])
                    Y.append(y_last[:, gap:, :])
                    LABELS.append(label_last[:, gap:, :])
            else:
                for gap in gaps:
                    X.append(x_batch_data[:, gap:, :])
                    Y.append(y_last[:, gap:, :])
                    LABELS.append(label_last[:, gap:, :])

            for i in range(N): generators[i].eval(); discriminators[i].train()
            loss_D, lossD_G = discriminate_fake(args, X, Y, LABELS, generators, discriminators, window_sizes,
                                                y_last.shape[-1], adv_criterion, reg_criterion, cls_criterion,
                                                weight_matrix, device, "train_D")
            for i in range(N): loss_dict[d_keys[i]].append(loss_D[i].item())
            for i in range(1, N + 1):
                for j in range(1, N + 1): loss_dict[f'D{i}_G{j}'].append(lossD_G[i - 1, j - 1].item())

            for opt in optimizers_D: opt.zero_grad()
            scaler.scale(loss_D.sum(dim=0)).backward()

            for i in range(N):
                scaler.step(optimizers_D[i])
            scaler.update()  # 对所有判别器执行step后，执行一次update

            for i in range(N):
                discriminators[i].eval()
                generators[i].train()

            weight = weight_matrix[:, :-1].clone().detach()
            loss_G, loss_mse_G = discriminate_fake(args, X, Y, LABELS, generators, discriminators, window_sizes,
                                                   y_last.shape[-1], adv_criterion, reg_criterion, cls_criterion,
                                                   weight, device, "train_G")
            for i in range(N): loss_dict[g_keys[i]].append(loss_G[i].item()); loss_dict["MSE_" + g_keys[i]].append(
                loss_mse_G[i].item())

            for opt in optimizers_G: opt.zero_grad()
            scaler.scale(loss_G.sum(dim=0)).backward()

            for opt in optimizers_G:
                scaler.step(opt)
            scaler.update()  # 对所有生成器执行step后，执行一次update

        for k, v in loss_dict.items(): hists_dict[k][epoch] = np.mean(v)

        improved = [False] * N
        for i in range(N):
            val_mse, val_acc, val_bce, val_cls_loss = validate_with_label(generators[i], discriminators[i], val_xes[i],
                                                                          val_y_gan[i], val_label_gan[i], adv_criterion,
                                                                          cls_criterion)
            hists_dict[val_loss_keys[i]][epoch], hists_dict[acc_keys[i]][epoch] = val_mse.item(), val_acc.item()
            if args.monitor_metric == 'val_mse':
                current_metric = val_mse.item()
            elif args.monitor_metric == 'val_acc':
                current_metric = val_acc.item()
            elif args.monitor_metric == 'val_bce':
                current_metric = val_bce.item()
            else:
                current_metric = val_cls_loss.item()
            is_better = (args.monitor_metric != 'val_acc' and current_metric < best_monitor_metric[i]) or (
                    args.monitor_metric == 'val_acc' and current_metric > best_monitor_metric[i])
            if is_better: best_monitor_metric[i] = current_metric; best_model_state[i] = copy.deepcopy(
                generators[i].state_dict()); best_epoch[i] = epoch + 1; improved[i] = True
            schedulers[i].step(current_metric)

        if (epoch + 1) % 10 == 0 and cross_finetune_epochs > 0:
            G_losses = [hists_dict[val_loss_keys[i]][epoch] for i in range(N)];
            D_losses = [hists_dict[d_keys[i]][epoch] for i in range(N)]
            G_rank, D_rank = np.argsort(G_losses), np.argsort(D_losses)
            print(f"--- Start Cross Finetune on Epoch {epoch + 1} ---");
            logging.info(f"Start cross finetune on Epoch {epoch + 1}! G{G_rank[0] + 1} with D{D_rank[0] + 1}")
            cross_best_Gloss = np.inf
            for e in range(cross_finetune_epochs):
                for batch_idx, (x_batch_data, y_last, label_last) in enumerate(dataloaders[-1]):
                    if label_last.dim() == 2: label_last = label_last.unsqueeze(-1)
                    x_batch_data, y_last, label_last = x_batch_data.to(device), y_last.to(device), label_last.to(device)

                    X_fine, Y_fine, LABELS_fine = [], [], []
                    if is_mpd_run:
                        for gap in gaps:
                            X_fine.append(x_batch_data[:, :, gap:, :]);
                            Y_fine.append(y_last[:, gap:, :]);
                            LABELS_fine.append(label_last[:, gap:, :])
                    else:
                        for gap in gaps:
                            X_fine.append(x_batch_data[:, gap:, :]);
                            Y_fine.append(y_last[:, gap:, :]);
                            LABELS_fine.append(label_last[:, gap:, :])

                    X_finetune, Y_finetune, LABELS_finetune = X_fine[G_rank[0]], Y_fine[D_rank[0]], LABELS_fine[
                        D_rank[0]]
                    generators[G_rank[0]].eval();
                    discriminators[D_rank[0]].train()
                    loss_D, _ = discriminate_fake(args, [X_finetune], [Y_finetune], [LABELS_finetune],
                                                  [generators[G_rank[0]]], [discriminators[D_rank[0]]],
                                                  [window_sizes[D_rank[0]]], y_last.shape[-1], adv_criterion,
                                                  reg_criterion, cls_criterion, weight_matrix[D_rank[0], G_rank[0]],
                                                  device, "train_D")

                    optimizers_D[D_rank[0]].zero_grad();
                    scaler.scale(loss_D.sum(dim=0)).backward();
                    scaler.step(optimizers_D[D_rank[0]]);
                    scaler.update()

                    discriminators[D_rank[0]].eval();
                    generators[G_rank[0]].train()
                    loss_G, _ = discriminate_fake(args, [X_finetune], [Y_finetune], [LABELS_finetune],
                                                  [generators[G_rank[0]]], [discriminators[D_rank[0]]],
                                                  [window_sizes[D_rank[0]]], y_last.shape[-1], adv_criterion,
                                                  reg_criterion, cls_criterion,
                                                  weight_matrix[D_rank[0], :-1].clone().detach()[G_rank[0]], device,
                                                  "train_G")

                    optimizers_G[G_rank[0]].zero_grad();
                    scaler.scale(loss_G.sum(dim=0)).backward();
                    scaler.step(optimizers_G[G_rank[0]]);
                    scaler.update()

                validate_G_loss, validate_G_acc, _, _ = validate_with_label(generators[G_rank[0]],
                                                                            discriminators[D_rank[0]],
                                                                            val_xes[G_rank[0]], val_y_gan[G_rank[0]],
                                                                            val_label_gan[G_rank[0]], adv_criterion,
                                                                            cls_criterion)
                logging.info(
                    f"Cross finetune Sub-Epoch [{e + 1}/{cross_finetune_epochs}]: MSE {validate_G_loss:.8f}, Acc {validate_G_acc * 100:.2f}%")
                if validate_G_loss >= cross_best_Gloss:
                    generators[G_rank[0]].load_state_dict(best_model_state[G_rank[0]])
                    break
                else:
                    cross_best_Gloss = validate_G_loss
                    best_model_state[G_rank[0]] = copy.deepcopy(generators[G_rank[0]].state_dict());
                    best_epoch[G_rank[0]] = epoch + 1
                    if args.monitor_metric == 'val_mse':
                        best_monitor_metric[G_rank[0]] = validate_G_loss.item()
                    elif args.monitor_metric == 'val_acc':
                        best_monitor_metric[G_rank[0]] = validate_G_acc.item()
            print(f"--- End Cross Finetune on Epoch {epoch + 1} ---")

        if distill_epochs > 0 and (epoch + 1) % 30 == 0:
            rank = np.argsort([hists_dict[val_loss_keys[i]][epoch] for i in range(N)])
            print(f"--- Start Distillation on Epoch {epoch + 1} ---");
            logging.info(f"Distill on Epoch {epoch + 1} from G{rank[0] + 1} to G{rank[-1] + 1}")
            for e in range(distill_epochs):
                # Pass the local scaler to the distill function
                do_distill(rank, generators, dataloaders, optimizers_G, window_sizes,
                           device, is_mpd_run, scaler=scaler)

        epoch_time = time.time() - epo_start
        log_mse_str = ", ".join(f"G{i + 1}: {hists_dict[k][epoch]:.8f}" for i, k in enumerate(val_loss_keys));
        log_acc_str = ", ".join(f"G{i + 1}: {hists_dict[k][epoch] * 100:.2f}%" for i, k in enumerate(acc_keys));
        best_epoch_str = ", ".join(f'G{i + 1}:{best_epoch[i]}' for i in range(N))
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Time: {epoch_time:.2f}s");
        print(f"  Validation MSE -> {log_mse_str} | Validation ACC -> {log_acc_str}");
        print(f"  Patience: {patience_counter} | Best Epochs: {best_epoch_str}")
        logging.info(f"EPOCH {epoch + 1} | Validation MSE: {log_mse_str} | Accuracy: {log_acc_str}")
        if not any(improved):
            patience_counter += 1
        else:
            patience_counter = 0
        if patience_counter >= patience: print("Early stopping triggered."); break

    data_G, data_D = [[[] for _ in range(N + 1)] for _ in range(N)], [[[] for _ in range(N + 1)] for _ in range(N)]
    for i in range(N):
        for j in range(N + 1):
            if j < N:
                data_G[i][j] = hists_dict[f"D{j + 1}_G{i + 1}"][:epoch + 1];
                data_D[i][j] = hists_dict[
                                   f"D{i + 1}_G{j + 1}"][
                               :epoch + 1]
            else:
                data_G[i][j] = hists_dict[g_keys[i]][:epoch + 1];
                data_D[i][j] = hists_dict[d_keys[i]][:epoch + 1]
    plot_generator_losses(data_G, output_dir);
    plot_discriminator_losses(data_D, output_dir);
    visualize_overall_loss([d[N] for d in data_G], [d[N] for d in data_D], output_dir)
    hist_MSE_G, hist_val_loss = [hists_dict[f"MSE_G{i + 1}"][:epoch + 1] for i in range(N)], [
        hists_dict[f"val_G{i + 1}"][:epoch + 1] for i in range(N)]
    plot_mse_loss(hist_MSE_G, hist_val_loss, epoch + 1, output_dir);
    logging.info(f"Best epochs | {', '.join(f'G{i + 1}:{best_epoch[i]}' for i in range(N))}")
    results = evaluate_best_models(generators, best_model_state, train_xes, train_y, val_xes, val_y, y_scaler,
                                   output_dir, window_sizes, date_series=date_series)
    return results, best_model_state


def discriminate_fake(args, X, Y, LABELS, generators, discriminators, window_sizes, target_num, adv_criterion,
                      reg_criterion, cls_criterion, weight_matrix, device, mode):
    assert mode in ["train_D", "train_G"];
    N = len(generators)

    with get_autocast_context(args.amp_dtype):
        dis_real_outputs = [m(y, l) for m, y, l in zip(discriminators, Y, LABELS)];
        outputs = [g(x) for g, x in zip(generators, X)]
        real_labels = [torch.ones_like(o).to(device) for o in dis_real_outputs];
        fake_data_G, fake_logits_G = zip(*outputs);
        fake_cls_G = [torch.argmax(logit, 1) for logit in fake_logits_G]
        lossD_real = [adv_criterion(o, r) for o, r in zip(dis_real_outputs, real_labels)]

    if mode == "train_D":
        fake_data_temp_G = [d.detach() for d in fake_data_G];
        fake_cls_temp_G = [c.detach() for c in fake_cls_G];
    elif mode == "train_G":
        fake_data_temp_G = fake_data_G
        fake_cls_temp_G = fake_cls_G

    fake_data_for_disc = [torch.cat([y[:, :ws, :], d.reshape(-1, 1, target_num)], 1) for y, ws, d in
                          zip(Y, window_sizes, fake_data_temp_G)]
    fake_labels_for_disc = [torch.cat([l[:, :ws, :], c.reshape(-1, 1, 1)], 1) for l, ws, c in
                            zip(LABELS, window_sizes, fake_cls_temp_G)]

    fake_data_GtoD, fake_cls_GtoD = {}, {}
    for i in range(N):
        for j in range(N):
            if i < j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = torch.cat(
                    [Y[j][:, :window_sizes[j] - window_sizes[i], :], fake_data_for_disc[i]], 1);
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = torch.cat(
                    [LABELS[j][:, :window_sizes[j] - window_sizes[i], :], fake_labels_for_disc[i]], 1)
            elif i > j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_for_disc[i][:, window_sizes[i] - window_sizes[j]:, :];
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_labels_for_disc[i][:, window_sizes[i] - window_sizes[j]:, :]
            else:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_for_disc[i];
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_labels_for_disc[i]

    fake_labels = [torch.zeros_like(r).to(device) for r in real_labels]
    with get_autocast_context(args.amp_dtype):
        dis_fake_outputD = []
        for i in range(N):
            row = [discriminators[i](fake_data_GtoD[f"G{j + 1}ToD{i + 1}"], fake_cls_GtoD[f"G{j + 1}ToD{i + 1}"].long())
                   for j in range(N)]
            if mode == "train_D":
                row.append(lossD_real[i])
            dis_fake_outputD.append(row)

        if mode == "train_D":
            loss_matrix = torch.zeros(N, N + 1, device=device);
            weight = weight_matrix.clone().detach()
            for i in range(N):
                for j in range(N + 1):
                    loss_matrix[i, j] = adv_criterion(dis_fake_outputD[i][j], fake_labels[i]) if j < N else \
                        dis_fake_outputD[i][j]
        elif mode == "train_G":
            loss_matrix = torch.zeros(N, N, device=device);
            weight = weight_matrix.clone().detach()
            for i in range(N):
                for j in range(N):
                    loss_matrix[i, j] = adv_criterion(dis_fake_outputD[i][j], real_labels[i])

        loss_DorG = torch.multiply(weight, loss_matrix).sum(dim=1)

        if mode == "train_G":
            loss_reg_G = [reg_criterion(d.squeeze(), y[:, -1, :].squeeze()) for d, y in zip(fake_data_G, Y)];
            loss_matrix_reg = loss_reg_G
            loss_DorG += torch.stack(loss_reg_G).to(device)
            cls_losses = [cls_criterion(c, l[:, -1, :].squeeze().long()) for c, l in zip(fake_logits_G, LABELS)];
            loss_DorG += torch.stack(cls_losses)
            return loss_DorG, loss_matrix_reg

    return loss_DorG, loss_matrix


def do_distill(rank, generators, dataloaders, optimizers, window_sizes, device, is_mpd_run, *, scaler, alpha=0.3,
               temperature=2.0,
               grad_clip=1.0, mse_lambda=0.8):
    teacher, student, opt = generators[rank[0]], generators[rank[-1]], optimizers[rank[-1]];
    teacher.eval();
    student.train()
    loader = dataloaders[rank[0]] if window_sizes[rank[0]] > window_sizes[rank[-1]] else dataloaders[rank[-1]]
    if not loader:
        logging.warning("蒸馏的Dataloader为空，跳过蒸馏。")
        return
    gap = abs(window_sizes[rank[0]] - window_sizes[rank[-1]])

    for x_batch, y, label in loader:
        y, label = y[:, -1, :].to(device), label[:, -1].to(device)

        if is_mpd_run:
            x_t = x_batch if window_sizes[rank[0]] > window_sizes[rank[-1]] else x_batch[:, :, gap:, :]
            x_s = x_batch[:, :, gap:, :] if window_sizes[rank[0]] > window_sizes[rank[-1]] else x_batch
        else:
            x_t = x_batch if window_sizes[rank[0]] > window_sizes[rank[-1]] else x_batch[:, gap:, :]
            x_s = x_batch[:, gap:, :] if window_sizes[rank[0]] > window_sizes[rank[-1]] else x_batch

        x_t, x_s = x_t.to(device), x_s.to(device)

        with torch.no_grad():
            t_out, t_cls = teacher(x_t)

        s_out, s_cls = student(x_s)

        soft_loss = F.kl_div(F.log_softmax(s_cls / temperature, 1), F.softmax(t_cls / temperature, 1),
                             reduction="batchmean") * (alpha * temperature ** 2)
        hard_loss = nn.CrossEntropyLoss()(s_cls, label.long()) * (1. - alpha) + F.mse_loss(s_out, y) * (
                1. - alpha) * mse_lambda
        distillation_loss = soft_loss + hard_loss

        opt.zero_grad();
        scaler.scale(distillation_loss).backward()
        if grad_clip: clip_grad_norm_(student.parameters(), grad_clip)
        scaler.step(opt);
        scaler.update()