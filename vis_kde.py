# 文件名: vis_kde.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import argparse
import glob
import numpy as np
import multiprocessing
import traceback
import logging
import re
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, mean_absolute_error, classification_report

# --- 全局字体设置 ---
plt.style.use('seaborn-v0_8-whitegrid')
try:
    plt.rcParams.update({'font.size': 14, 'font.family': 'SimHei', 'axes.unicode_minus': False})
    logging.info("成功设置字体为 SimHei。")
except Exception:
    plt.rcParams.update({'font.size': 14, 'axes.unicode_minus': False})
    logging.warning("警告: 未找到 'SimHei' 字体。图表中的中文可能无法正常显示。")


# --- 辅助函数 ---
def get_model_info_from_filename(csv_path, all_window_sizes):
    filename = os.path.basename(csv_path)
    match = re.search(r'predictions_gen_(\d+)_(\w+)\.csv', filename)
    if not match:
        logging.warning(f"无法从文件名 {filename} 中解析模型信息。")
        return None, None
    gen_index = int(match.group(1)) - 1
    model_type = match.group(2).upper()
    if gen_index < len(all_window_sizes):
        window_size = all_window_sizes[gen_index]
        model_label = f"G{gen_index + 1} - {model_type} (Win={window_size})"
        return model_label, gen_index
    else:
        logging.warning(f"索引 {gen_index} 超出窗口大小列表范围。")
        model_label = f"G{gen_index + 1} - {model_type}"
        return model_label, gen_index


# --- 绘图函数 ---
def plot_kde_for_experiment(model_data, output_path, stock_name, sector_name, alpha, no_grid):
    plt.figure(figsize=(14, 8));
    all_true_series = []
    for data in model_data:
        model_label, df = data['label'], data['df']
        if 'true' not in df.columns or 'pred' not in df.columns: continue
        sns.kdeplot(df['pred'].dropna(), label=f'预测值 ({model_label})', linewidth=1.5, alpha=alpha, fill=True)
        if not all_true_series: all_true_series.append(df['true'].dropna())
    if all_true_series:
        combined_true = pd.concat(all_true_series).dropna()
        if not combined_true.empty: sns.kdeplot(combined_true, label='真实值 (统一)', color='orangered', linewidth=2.5,
                                                fill=True, alpha=max(0, alpha - 0.1), zorder=0)
    plt.title(f'真实值 vs. 预测值 KDE 分布\n({sector_name} - {stock_name})', fontsize=20, pad=20);
    plt.xlabel('收盘价', fontsize=16);
    plt.ylabel('密度', fontsize=16);
    plt.legend(title='数据来源', fontsize=12)
    if no_grid: plt.grid(False)
    plt.tight_layout();
    os.makedirs(os.path.dirname(output_path), exist_ok=True);
    plt.savefig(output_path, dpi=300);
    plt.close();
    logging.info(f"成功保存KDE图表: {output_path}")


def plot_scatter_for_generator(df, output_path, model_label, stock_name, sector_name, no_grid):
    plt.figure(figsize=(10, 10));
    plt.scatter(df['true'], df['pred'], alpha=0.5, s=15, label='预测点');
    min_val, max_val = min(df['true'].min(), df['pred'].min()), max(df['true'].max(), df['pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想情况 (y=x)');
    r_squared = r2_score(df['true'], df['pred'])
    plt.title(f'{model_label} 预测散点图\n({sector_name} - {stock_name})', fontsize=18, pad=20);
    plt.xlabel('真实收盘价', fontsize=14);
    plt.ylabel('预测收盘价', fontsize=14);
    plt.legend()
    plt.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$', transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    if no_grid: plt.grid(False)
    plt.axis('equal');
    plt.tight_layout();
    os.makedirs(os.path.dirname(output_path), exist_ok=True);
    plt.savefig(output_path, dpi=300);
    plt.close();
    logging.info(f"成功保存散点图: {output_path}")


def plot_timeseries_for_generator(df, output_path, model_label, stock_name, sector_name, no_grid):
    plt.figure(figsize=(16, 8))
    if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
        plt.plot(df['date'], df['true'], label='真实值', color='royalblue', linewidth=2);
        plt.plot(df['date'], df['pred'], label='预测值', color='darkorange', linestyle='--', linewidth=1.5)
        ax = plt.gca();
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'));
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12));
        plt.gcf().autofmt_xdate();
        plt.xlabel('日期', fontsize=14)
    else:
        plt.plot(df.index, df['true'], label='真实值', color='royalblue', linewidth=2); plt.plot(df.index, df['pred'],
                                                                                                 label='预测值',
                                                                                                 color='darkorange',
                                                                                                 linestyle='--',
                                                                                                 linewidth=1.5); plt.xlabel(
            '时间步', fontsize=14)
    plt.title(f'{model_label} 时序预测对比\n({sector_name} - {stock_name})', fontsize=18, pad=20);
    plt.ylabel('收盘价', fontsize=14);
    plt.legend();
    plt.grid(True, which='both', linestyle='--', linewidth=0.5);
    plt.tight_layout();
    os.makedirs(os.path.dirname(output_path), exist_ok=True);
    plt.savefig(output_path, dpi=300);
    plt.close();
    logging.info(f"成功保存时序图: {output_path}")


def plot_residuals_vs_fitted(df, output_path, model_label, stock_name, sector_name, no_grid):
    df['residuals'] = df['true'] - df['pred'];
    plt.figure(figsize=(12, 7));
    sns.residplot(x=df['pred'], y=df['residuals'], lowess=True, scatter_kws={'alpha': 0.5},
                  line_kws={'color': 'red', 'lw': 2, 'alpha': 0.8})
    plt.title(f'{model_label} 残差 vs. 预测值图\n({sector_name} - {stock_name})', fontsize=18, pad=20);
    plt.xlabel('预测收盘价 (Fitted Values)', fontsize=14);
    plt.ylabel('残差 (True - Pred)', fontsize=14)
    if no_grid: plt.grid(False); plt.tight_layout(); os.makedirs(os.path.dirname(output_path),
                                                                 exist_ok=True); plt.savefig(output_path,
                                                                                             dpi=300); plt.close(); logging.info(
        f"成功保存残差vs.预测值图: {output_path}")


def plot_confusion_matrix(df_class, output_path, model_label, stock_name, sector_name):
    labels = [0, 1, 2];
    cm = confusion_matrix(df_class['true_action'], df_class['predicted_action'], labels=labels);
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['预测跌', '预测平', '预测涨'],
                yticklabels=['真实跌', '真实平', '真实涨'])
    plt.title(f'{model_label} 分类混淆矩阵\n({sector_name} - {stock_name})', fontsize=18, pad=20);
    plt.ylabel('真实类别', fontsize=14);
    plt.xlabel('预测类别', fontsize=14)
    plt.tight_layout();
    os.makedirs(os.path.dirname(output_path), exist_ok=True);
    plt.savefig(output_path, dpi=300);
    plt.close();
    logging.info(f"成功保存混淆矩阵: {output_path}")


def plot_classification_metrics_bar(model_metrics, output_path, stock_name, sector_name):
    if not model_metrics:
        logging.warning("没有可用于绘制分类指标图的数据。")
        return
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    data_fall = {metric: [d.get('0', {}).get(metric, 0) for d in model_metrics.values()] for metric in metrics_to_plot}
    data_rise = {metric: [d.get('2', {}).get(metric, 0) for d in model_metrics.values()] for metric in metrics_to_plot}
    model_labels = list(model_metrics.keys())
    x = np.arange(len(model_labels));
    width = 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7));
    fig.suptitle(f'各模型分类性能对比\n({sector_name} - {stock_name})', fontsize=20, y=1.02)
    rects1_fall = ax1.bar(x - width, data_fall['precision'], width, label='Precision');
    rects2_fall = ax1.bar(x, data_fall['recall'], width, label='Recall');
    rects3_fall = ax1.bar(x + width, data_fall['f1-score'], width, label='F1-Score')
    ax1.set_ylabel('分数');
    ax1.set_title('“预测跌” (类别 0) 性能');
    ax1.set_xticks(x);
    ax1.set_xticklabels(model_labels, rotation=30, ha='right');
    ax1.legend();
    ax1.bar_label(rects1_fall, padding=3, fmt='%.2f');
    ax1.bar_label(rects2_fall, padding=3, fmt='%.2f');
    ax1.bar_label(rects3_fall, padding=3, fmt='%.2f');
    ax1.set_ylim(0, 1.1)
    rects1_rise = ax2.bar(x - width, data_rise['precision'], width, label='Precision');
    rects2_rise = ax2.bar(x, data_rise['recall'], width, label='Recall');
    rects3_rise = ax2.bar(x + width, data_rise['f1-score'], width, label='F1-Score')
    ax2.set_ylabel('分数');
    ax2.set_title('“预测涨” (类别 2) 性能');
    ax2.set_xticks(x);
    ax2.set_xticklabels(model_labels, rotation=30, ha='right');
    ax2.legend();
    ax2.bar_label(rects1_rise, padding=3, fmt='%.2f');
    ax2.bar_label(rects2_rise, padding=3, fmt='%.2f');
    ax2.bar_label(rects3_rise, padding=3, fmt='%.2f');
    ax2.set_ylim(0, 1.1)
    fig.tight_layout();
    os.makedirs(os.path.dirname(output_path), exist_ok=True);
    plt.savefig(output_path, dpi=300);
    plt.close();
    logging.info(f"成功保存分类性能对比图: {output_path}")


# --- 主处理函数 ---
def process_single_experiment(exp_dir, output_dir_base, alpha, no_grid, all_window_sizes):
    try:
        true2pred_dir = os.path.join(exp_dir, 'true2pred_csv')
        if not os.path.isdir(true2pred_dir):
            logging.warning(f"在 {exp_dir} 中未找到 'true2pred_csv' 目录，跳过。")
            return False
        csv_files = glob.glob(os.path.join(true2pred_dir, '*.csv'))
        if not csv_files:
            logging.warning(f"目录 {true2pred_dir} 中未找到 CSV 文件，跳过。")
            return False

        path_parts = exp_dir.replace('\\', '/').split('/');
        stock_name = path_parts[-1];
        sector_name = path_parts[-2]
        logging.info(f"开始处理股票: {sector_name} - {stock_name} (目录: {exp_dir})")

        stock_vis_output_dir = os.path.join(output_dir_base, sector_name, stock_name)
        dir_kde = os.path.join(stock_vis_output_dir, '1_KDE_Distribution');
        dir_timeseries = os.path.join(stock_vis_output_dir, '2_Timeseries_Plots');
        dir_scatter = os.path.join(stock_vis_output_dir, '3_Scatter_Plots');
        dir_residuals = os.path.join(stock_vis_output_dir, '4_Residuals_Plots');
        dir_cm = os.path.join(stock_vis_output_dir, '5_Confusion_Matrices');
        dir_metrics = os.path.join(stock_vis_output_dir, '6_Classification_Metrics')

        model_data_list = []
        for csv_path in sorted(csv_files):
            model_label, gen_index = get_model_info_from_filename(csv_path, all_window_sizes)
            if not model_label: continue
            df = pd.read_csv(csv_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.dropna(subset=['date'], inplace=True)
            model_data_list.append({'label': model_label, 'df': df, 'gen_index': gen_index})

        if not model_data_list:
            logging.warning(f"股票 {sector_name} - {stock_name} 未能加载任何有效的模型数据。")
            return False

        kde_path = os.path.join(dir_kde, 'all_models_kde.png')
        plot_kde_for_experiment(model_data_list, kde_path, stock_name, sector_name, alpha, no_grid)

        all_models_metrics = {}

        for data in model_data_list:
            model_label, df, gen_index = data['label'], data['df'], data['gen_index']
            safe_model_filename = re.sub(r'[^\w\-.()=]', '', model_label.replace(' ', '_')) + '.png'

            try:
                plot_timeseries_for_generator(df, os.path.join(dir_timeseries, safe_model_filename), model_label,
                                              stock_name, sector_name, no_grid)
                plot_scatter_for_generator(df, os.path.join(dir_scatter, safe_model_filename), model_label, stock_name,
                                           sector_name, no_grid)
                plot_residuals_vs_fitted(df, os.path.join(dir_residuals, safe_model_filename), model_label, stock_name,
                                         sector_name, no_grid)

                signal_files = glob.glob(os.path.join(exp_dir, f'G{gen_index + 1}_*_daily_signals.csv'))
                if signal_files:
                    df_signals = pd.read_csv(signal_files[0])
                    df_signals['date'] = pd.to_datetime(df_signals['date'], format='%Y%m%d', errors='coerce')
                    df_signals.dropna(subset=['date'], inplace=True)
                    df_merged = pd.merge(df, df_signals[['date', 'predicted_action']], on='date', how='inner')

                    if df_merged.empty:
                        logging.warning(
                            f"模型 {model_label}: 'true2pred'和'daily_signals'数据日期无交集，无法生成分类图表。")
                        continue

                    # --- 优化点：一次性准备好分类数据 ---
                    df_class_data = df_merged.copy()
                    df_class_data['true_action'] = 1
                    df_class_data.loc[df_class_data['true'] > df_class_data['true'].shift(1), 'true_action'] = 2
                    df_class_data.loc[df_class_data['true'] < df_class_data['true'].shift(1), 'true_action'] = 0
                    df_class_data.dropna(subset=['true_action', 'predicted_action'], inplace=True)

                    if not df_class_data.empty:
                        plot_confusion_matrix(df_class_data, os.path.join(dir_cm, safe_model_filename), model_label,
                                              stock_name, sector_name)
                        report = classification_report(df_class_data['true_action'], df_class_data['predicted_action'],
                                                       labels=[0, 2], output_dict=True, zero_division=0)
                        all_models_metrics[model_label] = report
                else:
                    logging.warning(f"未找到模型 {model_label} 对应的每日信号文件，跳过分类图表生成。")

            except Exception as e:
                logging.error(f"为模型 {model_label} 生成图表时发生错误: {e}\n{traceback.format_exc()}")

        metrics_bar_path = os.path.join(dir_metrics, 'precision_recall_f1_comparison.png')
        plot_classification_metrics_bar(all_models_metrics, metrics_bar_path, stock_name, sector_name)

        logging.info(f"完成处理股票: {sector_name} - {stock_name}")
        return True
    except Exception as e:
        logging.error(f"处理实验目录 {exp_dir} 时发生未预期错误: {e}\n{traceback.format_exc()}")
        return False


# --- 主控函数 ---
def find_and_process_experiments(root_dir: str, output_dir: str, alpha: float, no_grid: bool, num_processes: int):
    all_stock_dirs = glob.glob(os.path.join(root_dir, '*', '*'), recursive=False)
    experiment_dirs_to_process = []
    for stock_dir in sorted(all_stock_dirs):
        if os.path.isdir(stock_dir) and os.path.isdir(os.path.join(stock_dir, 'true2pred_csv')):
            experiment_dirs_to_process.append(stock_dir)
    if not experiment_dirs_to_process:
        print(f"错误: 在根目录 '{root_dir}' 下未找到任何有效的股票实验结果目录。");
        return
    print(f"找到 {len(experiment_dirs_to_process)} 个股票实验结果目录待处理，开始并行生成图表...")

    first_exp_dir = experiment_dirs_to_process[0]
    args_log_path = os.path.join(first_exp_dir, 'logs')
    log_files = glob.glob(os.path.join(args_log_path, '*.log'))
    all_window_sizes = [5, 10, 15]
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        with open(latest_log, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            match = re.search(r'"window_sizes":\s*\[([\d,\s]+)\]', first_line)
            if match:
                all_window_sizes = [int(x.strip()) for x in match.group(1).split(',')]
                logging.info(f"从日志中成功解析到窗口大小: {all_window_sizes}")
    else:
        logging.warning("未找到日志文件，将使用默认窗口大小 [5, 10, 15]。")

    tasks_to_run = []
    for exp_dir in experiment_dirs_to_process:
        path_parts = exp_dir.replace('\\', '/').split('/');
        stock_name = path_parts[-1];
        sector_name = path_parts[-2]
        check_dir = os.path.join(output_dir, sector_name, stock_name, '1_KDE_Distribution')
        if os.path.isdir(check_dir) and os.listdir(check_dir):
            logging.info(f"股票 {sector_name} - {stock_name} 的图表目录已存在，跳过。");
            continue
        tasks_to_run.append((exp_dir, output_dir, alpha, no_grid, all_window_sizes))

    if not tasks_to_run:
        print("所有股票的图表均已存在，无需生成新的图表。");
        return

    results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        async_results = pool.starmap_async(process_single_experiment, tasks_to_run)
        success_count = sum(1 for res in async_results.get() if res)
        failure_count = len(tasks_to_run) - success_count
        results.append(f"总共成功处理 {success_count} 个实验，失败 {failure_count} 个。")

    print("\n--- 图表生成处理总结 ---")
    for result in results: print(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description='为每个实验（股票）的真实值和预测值绘制多种评估图表。')
    parser.add_argument('--input_dir', type=str, default='output/multi_gan',
                        help='包含所有实验结果的根目录 (例如 "output/multi_gan")。')
    parser.add_argument('--output_dir', type=str, default='output_vis', help='保存生成的所有评估图表的根目录。')
    parser.add_argument('--alpha', type=float, default=0.4, help='KDE和直方图填充区域的透明度。')
    parser.add_argument('--no_grid', action='store_true', help='添加此参数以移除图表中的网格线。')
    parser.add_argument('--num_processes', type=int, default=multiprocessing.cpu_count(),
                        help=f"指定用于并行生成图表的CPU核心数。默认为系统核心数 ({multiprocessing.cpu_count()})。")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    find_and_process_experiments(args.input_dir, args.output_dir, args.alpha, args.no_grid, args.num_processes)
    print("\n所有评估图表生成任务已全部完成。")