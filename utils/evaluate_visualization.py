# 文件名: utils/evaluate_visualization.py

import torch
import torch.nn.functional as F
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import pandas as pd


# --- 验证函数 ---
def validate(model, val_x, val_y):
    model.eval()
    with torch.no_grad():
        device = val_x.device
        if isinstance(val_y, np.ndarray):
            val_y = torch.tensor(val_y, dtype=torch.float32, device=device)
        else:
            val_y = val_y.clone().detach().float().to(device)
        predictions, _ = model(val_x)
        mse_loss = F.mse_loss(predictions.squeeze(), val_y.squeeze())
        return mse_loss


def validate_with_label(generator, discriminator, val_x, val_y_gan, val_label_gan, adv_criterion):
    """
    验证模型在验证集上的MSE、分类准确率和对抗损失。
    修复了 val_label_gan 的维度问题。
    """
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        device = val_x.device

        # 1. 计算 MSE 和 ACC
        predictions, logits = generator(val_x)

        true_y_for_mse = val_y_gan[:, -1, :].squeeze()
        mse_loss = F.mse_loss(predictions.squeeze(), true_y_for_mse)

        # 使用二维索引
        true_cls_for_acc = val_label_gan[:, -1].squeeze()
        pred_cls = logits.argmax(dim=1)
        acc = (pred_cls == true_cls_for_acc).float().mean()

        # 2. 计算对抗损失
        target_num = val_y_gan.shape[-1]
        fake_data_for_disc = torch.cat([val_y_gan[:, :-1, :], predictions.reshape(-1, 1, target_num)], axis=1)

        # 调整 fake_labels_for_disc 的构造
        fake_labels_for_disc = torch.cat([val_label_gan[:, :-1], pred_cls.reshape(-1, 1)], axis=1)

        disc_output_on_fake = discriminator(fake_data_for_disc, fake_labels_for_disc.long())

        real_labels_for_loss = torch.ones_like(disc_output_on_fake).to(device)
        adversarial_loss = adv_criterion(disc_output_on_fake, real_labels_for_loss)

        return mse_loss, acc, adversarial_loss


# --- 绘图函数 ---
def plot_fitting_curve(true_values, predicted_values, dates, output_dir, model_name):
    viz_output_dir = os.path.join(output_dir, "visualization")
    os.makedirs(viz_output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    try:
        plt.rcParams.update({'font.size': 12, 'font.family': 'SimHei'})
    except:
        plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(15, 7))
    is_date_plotting = dates is not None and not dates.empty
    if is_date_plotting:
        x_axis_indices = np.arange(len(true_values))
        plt.plot(x_axis_indices, true_values, label='真实值', linewidth=2, color='royalblue', linestyle='-')
        plt.plot(x_axis_indices, predicted_values, label='预测值', linewidth=1.5, color='darkorange', linestyle='--')
        num_ticks = 10
        tick_positions = np.linspace(0, len(x_axis_indices) - 1, num_ticks, dtype=int)
        if not isinstance(dates, pd.Series):
            dates = pd.Series(dates)
        tick_labels = dates.iloc[tick_positions].dt.strftime('%Y-%m-%d')
        plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=30, ha='right')
        plt.xlabel('日期', fontsize=14)
    else:
        plt.plot(range(len(true_values)), true_values, label='真实值', linewidth=2, color='royalblue')
        plt.plot(range(len(predicted_values)), predicted_values, label='预测值', linewidth=1.5, color='darkorange',
                 linestyle='--')
        plt.xlabel('时间步', fontsize=14)
    plt.title(f'{model_name} 拟合曲线', fontsize=18);
    plt.ylabel('值', fontsize=14);
    plt.legend();
    plt.grid(True, which='both', linestyle='--', linewidth=0.5);
    plt.tight_layout()
    save_path = os.path.join(viz_output_dir, f'{model_name}_fitting_curve.png');
    plt.savefig(save_path, dpi=300);
    plt.close()


def compute_metrics(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - predicted_values) / (true_values + 1e-8))) * 100
    per_target_mse = np.mean((true_values - predicted_values) ** 2, axis=0)
    return mse, mae, rmse, mape, per_target_mse


def evaluate_best_models(generators, best_model_state,
                         train_xes, train_y,
                         eval_xes, eval_y,
                         y_scaler, output_dir,
                         window_sizes,
                         date_series=None):
    N = len(generators)
    for i in range(N):
        if best_model_state[i] is None:
            logging.warning(f"G{i + 1} 没有找到最佳模型状态，将跳过评估。")
            continue
        generators[i].load_state_dict(best_model_state[i])
        generators[i].eval()
    train_y_inv = y_scaler.inverse_transform(train_y.cpu().numpy().reshape(-1, 1)).flatten()
    eval_y_inv = y_scaler.inverse_transform(eval_y.cpu().numpy().reshape(-1, 1)).flatten()
    train_dates, eval_dates = None, None
    if date_series is not None and isinstance(date_series, pd.Series):
        max_window_size = max(window_sizes) if window_sizes else 0
        date_series_aligned = date_series.iloc[max_window_size:].reset_index(drop=True)
        train_size_seq, eval_size_seq = len(train_y), len(eval_y)
        if len(date_series_aligned) >= (train_size_seq + eval_size_seq):
            train_dates = date_series_aligned.iloc[:train_size_seq]
            eval_dates = date_series_aligned.iloc[train_size_seq: train_size_seq + eval_size_seq]
        else:
            logging.warning("对齐后的日期序列与数据长度不匹配，无法为绘图分配日期。")
    train_preds_inv, eval_preds_inv = [], [];
    train_metrics_list, eval_metrics_list = [], []
    with torch.no_grad():
        for i in range(N):
            if best_model_state[i] is None: continue
            train_pred, _ = generators[i](train_xes[i])
            train_pred_inv_i = y_scaler.inverse_transform(train_pred.cpu().numpy()).flatten()
            train_preds_inv.append(train_pred_inv_i)
            true_vals_for_metric = train_y_inv[-len(train_pred_inv_i):]
            train_metrics = compute_metrics(true_vals_for_metric, train_pred_inv_i)
            train_metrics_list.append(train_metrics)
            dates_for_plot = None
            if train_dates is not None and len(train_dates) >= len(train_pred_inv_i):
                dates_for_plot = train_dates.iloc[-len(train_pred_inv_i):]
            plot_fitting_curve(true_vals_for_metric, train_pred_inv_i, dates_for_plot, output_dir, f'G{i + 1}_Train')
            logging.info(
                f"Train Metrics for G{i + 1}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")
        for i in range(N):
            if best_model_state[i] is None: continue
            eval_pred, _ = generators[i](eval_xes[i])
            eval_pred_inv_i = y_scaler.inverse_transform(eval_pred.cpu().numpy()).flatten()
            eval_preds_inv.append(eval_pred_inv_i)
            true_vals_for_metric = eval_y_inv[-len(eval_pred_inv_i):]
            eval_metrics = compute_metrics(true_vals_for_metric, eval_pred_inv_i)
            eval_metrics_list.append(eval_metrics)
            dates_for_plot = None
            if eval_dates is not None and len(eval_dates) >= len(eval_pred_inv_i):
                dates_for_plot = eval_dates.iloc[-len(eval_pred_inv_i):]
            plot_fitting_curve(true_vals_for_metric, eval_pred_inv_i, dates_for_plot, output_dir, f'G{i + 1}_Test')
            logging.info(
                f"Test/Eval Metrics for G{i + 1}: MSE={eval_metrics[0]:.4f}, MAE={eval_metrics[1]:.4f}, RMSE={eval_metrics[2]:.4f}, MAPE={eval_metrics[3]:.4f}")
    while len(train_metrics_list) < N: train_metrics_list.append((np.nan,) * 5)
    while len(eval_metrics_list) < N: eval_metrics_list.append((np.nan,) * 5)
    result = {"train_mse": [m[0] for m in train_metrics_list], "train_mae": [m[1] for m in train_metrics_list],
              "train_rmse": [m[2] for m in train_metrics_list], "train_mape": [m[3] for m in train_metrics_list],
              "train_mse_per_target": [m[4] for m in train_metrics_list], "test_mse": [m[0] for m in eval_metrics_list],
              "test_mae": [m[1] for m in eval_metrics_list], "test_rmse": [m[2] for m in eval_metrics_list],
              "test_mape": [m[3] for m in eval_metrics_list], "test_mse_per_target": [m[4] for m in eval_metrics_list]}
    return result


def plot_generator_losses(data_G, output_dir):
    viz_output_dir = os.path.join(output_dir, "visualization");
    os.makedirs(viz_output_dir, exist_ok=True);
    plt.rcParams.update({'font.size': 12});
    all_data = data_G;
    N = len(all_data);
    num_losses_per_g = len(all_data[0]) if all_data else 0;
    plt.figure(figsize=(6 * N, 5))
    for i, data in enumerate(all_data):
        plt.subplot(1, N, i + 1)
        for j, acc in enumerate(data):
            label = f"G{i + 1} Combined" if j == num_losses_per_g - 1 else f"G{i + 1} vs D{j + 1}";
            plt.plot(acc, label=label, linewidth=2)
        plt.xlabel("Epoch", fontsize=14);
        plt.ylabel("Loss", fontsize=14);
        plt.title(f"G{i + 1} Loss over Epochs", fontsize=16);
        plt.legend();
        plt.grid(True)
    plt.tight_layout();
    plt.savefig(os.path.join(viz_output_dir, "generator_losses.png"), dpi=300);
    plt.close()


def plot_discriminator_losses(data_D, output_dir):
    viz_output_dir = os.path.join(output_dir, "visualization");
    os.makedirs(viz_output_dir, exist_ok=True);
    plt.rcParams.update({'font.size': 12});
    N = len(data_D);
    num_losses_per_d = len(data_D[0]) if data_D else 0;
    plt.figure(figsize=(6 * N, 5))
    for i, data in enumerate(data_D):
        plt.subplot(1, N, i + 1)
        for j, acc in enumerate(data):
            label = f"D{i + 1} Combined" if j == num_losses_per_d - 1 else f"D{i + 1} vs G{j + 1}";
            plt.plot(acc, label=label, linewidth=2)
        plt.xlabel("Epoch", fontsize=14);
        plt.ylabel("Loss", fontsize=14);
        plt.title(f"D{i + 1} Loss over Epochs", fontsize=16);
        plt.legend();
        plt.grid(True)
    plt.tight_layout();
    plt.savefig(os.path.join(viz_output_dir, "discriminator_losses.png"), dpi=300);
    plt.close()


def visualize_overall_loss(histG, histD, output_dir):
    viz_output_dir = os.path.join(output_dir, "visualization");
    os.makedirs(viz_output_dir, exist_ok=True);
    plt.rcParams.update({'font.size': 12});
    N = len(histG);
    plt.figure(figsize=(5 * N if N > 1 else 6, 5))
    for i, (g, d) in enumerate(zip(histG, histD)):
        plt.plot(g, label=f"G{i + 1} Loss", linewidth=2);
        plt.plot(d, label=f"D{i + 1} Loss", linewidth=2, linestyle='--')
    plt.xlabel("Epoch", fontsize=14);
    plt.ylabel("Loss", fontsize=14);
    plt.title("Overall Generator & Discriminator Loss", fontsize=16);
    plt.legend();
    plt.grid(True);
    plt.tight_layout();
    plt.savefig(os.path.join(viz_output_dir, "overall_losses.png"), dpi=300);
    plt.close()


def plot_mse_loss(hist_MSE_G, hist_val_loss, num_epochs, output_dir):
    viz_output_dir = os.path.join(output_dir, "visualization");
    os.makedirs(viz_output_dir, exist_ok=True);
    plt.rcParams.update({'font.size': 12});
    N = len(hist_MSE_G);
    plt.figure(figsize=(5 * N if N > 1 else 8, 5))
    for i, (MSE, val_loss) in enumerate(zip(hist_MSE_G, hist_val_loss)):
        plt.plot(range(num_epochs), MSE, label=f"Train MSE G{i + 1}", linewidth=2);
        plt.plot(range(num_epochs), val_loss, label=f"Val MSE G{i + 1}", linewidth=2, linestyle="--")
    plt.title("MSE Loss for Generators (Train vs Validation)", fontsize=16);
    plt.xlabel("Epoch", fontsize=14);
    plt.ylabel("MSE", fontsize=14);
    plt.legend();
    plt.grid(True);
    plt.tight_layout();
    plt.savefig(os.path.join(viz_output_dir, "mse_losses.png"), dpi=300);
    plt.close()