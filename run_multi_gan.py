# 文件名: run_multi_gan.py

import argparse
import pandas as pd
import os
import glob
import time
from utils.logger import setup_experiment_logging
from time_series_maa import MAA_time_series
import torch
import sys


def find_stock_files(base_dir, stock_name=None, sector=None):
    search_pattern = os.path.join(base_dir, 'train', '**', '*.csv')
    all_files = glob.glob(search_pattern, recursive=True)
    found_files = []
    if not stock_name and not sector:
        return all_files
    for f in all_files:
        path_parts = f.replace('\\', '/').split('/')
        if len(path_parts) < 3:
            continue
        file_stock_name = path_parts[-2]
        file_sector = path_parts[-3]
        if stock_name and file_stock_name == stock_name:
            found_files.append(f)
        elif sector and file_sector == sector:
            found_files.append(f)
    return found_files


def run_experiment_for_stock(args, stock_csv_path):
    path_parts = stock_csv_path.replace('\\', '/').split('/')
    stock_name = path_parts[-2]
    sector_name = path_parts[-3]

    # 统一的实验根目录，例如 output/multi_gan/板块/个股/
    experiment_base_dir = os.path.join(args.output_dir, sector_name, stock_name)
    # 正式训练的模型保存在其下的 ckpt 子目录
    stock_specific_ckpt_dir = os.path.join(experiment_base_dir, 'ckpt')
    # 其他输出（如日志、可视化图表）直接放在实验根目录
    stock_specific_output_dir = experiment_base_dir

    os.makedirs(stock_specific_output_dir, exist_ok=True)
    os.makedirs(stock_specific_ckpt_dir, exist_ok=True)

    print(f"\n{'=' * 20} 开始处理股票: {sector_name} - {stock_name} {'=' * 20}")
    print(f"数据源: {stock_csv_path}")
    print(f"统一实验目录: {stock_specific_output_dir}")
    print(f"模型保存目录: {stock_specific_ckpt_dir}")

    local_args = argparse.Namespace(**vars(args))
    local_args.output_dir = stock_specific_output_dir
    local_args.ckpt_dir = stock_specific_ckpt_dir

    # 实例化 MAA_time_series
    gca = MAA_time_series(local_args,
                          local_args.N_pairs, local_args.batch_size, local_args.num_epochs,
                          local_args.generators, local_args.discriminators,
                          stock_specific_ckpt_dir, stock_specific_output_dir,
                          local_args.window_sizes,
                          ckpt_path=local_args.ckpt_path,
                          initial_learning_rate=local_args.lr,
                          do_distill_epochs=local_args.distill_epochs,
                          cross_finetune_epochs=local_args.cross_finetune_epochs,
                          device=local_args.device,
                          seed=local_args.random_seed)

    # ==================== 新增: 个股CAE预训练触发逻辑 ====================
    if 'mpd' in local_args.generators and local_args.pretrain_cae:
        # CAE权重路径现在是相对于个股的输出目录
        cae_ckpt_path_stock = os.path.join(stock_specific_output_dir, local_args.cae_ckpt_filename)
        if os.path.exists(cae_ckpt_path_stock):
            print(f"\n--- 已找到该股票的预训练CAE权重 '{cae_ckpt_path_stock}'，跳过预训练。 ---")
        else:
            print(f"\n--- 未找到该股票的预训练CAE权重，将仅使用当前股票数据进行预训练... ---")
            # 调用预训练方法，只传入当前股票的CSV文件
            gca.pretrain_cae_if_needed(
                all_stock_files=[stock_csv_path],  # 只用自己的数据
                cae_ckpt_path=cae_ckpt_path_stock,
                pretrain_epochs=local_args.pretrain_cae_epochs
            )
            print(f"--- {stock_name} 的CAE预训练完成。权重已保存至 '{cae_ckpt_path_stock}'。 ---")
        # 将动态生成的、个股专属的CAE权重路径更新到参数中，以便后续加载
        local_args.cae_ckpt_path = cae_ckpt_path_stock
    # ========================== 预训练逻辑结束 ==========================

    # --- 后续流程使用更新后的 local_args ---

    # 加载和处理数据
    full_df_path = stock_csv_path
    full_df = pd.read_csv(full_df_path, usecols=['date'])
    date_series = pd.to_datetime(full_df['date'], format='%Y%m%d')
    predict_csv_path = stock_csv_path.replace(os.path.join('train'), os.path.join('predict'))
    gca.process_data(
        train_csv_path=stock_csv_path,
        predict_csv_path=predict_csv_path,
        target_column='close',
        exclude_columns=['date', 'direction']
    )

    # 初始化数据加载器和模型（init_model会在这里加载预训练权重）
    gca.init_dataloader()
    gca.init_model(local_args.num_classes)

    logger = setup_experiment_logging(stock_specific_output_dir, vars(local_args), f"train_{stock_name}")

    results = None
    if local_args.mode == "train":
        results, best_model_state = gca.train(logger, date_series=date_series)

        if best_model_state and any(s is not None for s in best_model_state):
            print("\n--- 训练结束，保存相关产物 ---")

            gca.save_models(best_model_state)
            gca.save_scalers()
            gca.generate_and_save_daily_signals(best_model_state, predict_csv_path)

            print("\n--- 加载最佳模型以生成预测对比CSV ---")
            for i in range(gca.N):
                if i < len(best_model_state) and best_model_state[i] is not None:
                    if i < len(gca.generators):
                        try:
                            gca.generators[i].load_state_dict(best_model_state[i])
                        except Exception as e:
                            print(f"警告: 加载生成器 G{i + 1} 状态用于 CSV 生成失败: {e}")
                    else:
                        print(f"警告: G{i + 1} 生成器未被初始化，无法加载状态用于 CSV 生成。")

            gca.save_predictions_to_csv(date_series=date_series)

    elif local_args.mode == "pred":
        results = gca.pred(date_series=date_series)

    if results:
        # 注意：这里的主结果文件还是保存在全局output_dir下，便于统一查看
        master_results_file = os.path.join(args.output_dir, "master_results.csv")
        timestamp_for_log = time.strftime("%Y%m%d-%H%M%S")
        result_row = {
            "timestamp": timestamp_for_log,
            "sector": sector_name,
            "stock": stock_name,
            "train_mse": results["train_mse"],
            "train_mae": results["train_mae"],
            "test_mse": results["test_mse"],
            "test_mae": results["test_mae"],
        }
        df = pd.DataFrame([result_row])
        header = not os.path.exists(master_results_file)
        df.to_csv(master_results_file, mode='a', header=header, index=False)
        print(f"===== 股票 {stock_name} 处理完毕, 结果已记录到 {master_results_file} =====")
    else:
        print(f"===== 股票 {stock_name} 处理完毕, 但没有结果需要记录 =====")