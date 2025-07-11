# 文件名: experiment_runner.py

import argparse
import torch
import os
import sys
import time
import pandas as pd
import logging
# 导入功能模块
import run_multi_gan

# ==============================================================================
# 默认参数配置中心 (COMMON_ARGS)
# ==============================================================================
COMMON_ARGS = {
    "stock_name": None,
    "sector": None,
    "data_base_dir": "csv_data",
    "output_dir": "output/multi_gan",
    "ckpt_dir": "ckpt/multi_gan",
    "notes": "Default run from COMMON_ARGS",
    "window_sizes": [5, 10, 15],
    "N_pairs": 3,
    "num_classes": 3,
    "generators": ["gru", "lstm", "transformer"],
    "discriminators": None,
    "distill_epochs": 1,
    "cross_finetune_epochs": 5,
    "num_epochs": 200,
    "lr": 2e-05,
    "batch_size": 64,
    "train_val_test_split": [0.7, 0.1, 0.2],
    "random_seed": 3407,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp_dtype": "none",
    "mode": "train",
    "ckpt_path": "auto",

    # 损失函数配置
    "adversarial_loss_mode": "bce",  # 可选 'bce', 'mse'
    "regression_loss_mode": "mse",  # 可选 'mse', 'mae'
    "monitor_metric": "val_mse",  # 可选 'val_mse', 'val_acc', 'val_bce'
}


# ==============================================================================
# setup_arg_parser 和 validate_args 定义在此处
# ==============================================================================

def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="为预处理好的股票数据运行Multi-GAN模型。这是实验的主控制脚本。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for key, value in COMMON_ARGS.items():
        if isinstance(value, list):
            parser.add_argument(f'--{key}', type=type(value[0]) if value else str, nargs='+', default=value,
                                help=f"默认为: {value}")
        elif isinstance(value, bool):
            if value is True:
                parser.add_argument(f'--{key}', action='store_false', default=value, help=f"设置此项以禁用 {key}")
            else:
                parser.add_argument(f'--{key}', action='store_true', default=value, help=f"设置此项以激活 {key}")
        else:
            arg_type = type(value) if value is not None else str
            parser.add_argument(f'--{key}', type=arg_type, default=value,
                                help=f"默认为: {value}")
    return parser


def validate_args(args):
    if len(args.generators) != args.N_pairs or len(args.window_sizes) != args.N_pairs:
        print("错误: --generators, --window_sizes, 和 --N_pairs 的数量必须一致!")
        sys.exit(1)

    if len(args.train_val_test_split) != 3 or not abs(sum(args.train_val_test_split) - 1.0) < 1e-6:
        print(f"错误: --train_val_test_split 必须包含3个数字且总和为1，当前为: {args.train_val_test_split}")
        sys.exit(1)

    if args.discriminators is None:
        args.discriminators = ["default"] * args.N_pairs
    elif len(args.discriminators) != args.N_pairs:
        print("错误: --discriminators 的数量必须与 --N_pairs 一致，或者为 None!")
        sys.exit(1)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"警告: 指定了设备 {args.device} 但CUDA不可用，将使用CPU。")
        args.device = "cpu"

    valid_amp_dtypes = ['none', 'float16', 'bfloat16', 'mixed']
    if args.amp_dtype.lower() not in valid_amp_dtypes:
        print(f"警告: 无效的 --amp_dtype 值 '{args.amp_dtype}'。可选值: {valid_amp_dtypes}。将使用 'none'。")
        args.amp_dtype = 'none'
    args.amp_dtype = args.amp_dtype.lower()

    valid_loss_modes = ['bce', 'mse', 'mae']
    if args.adversarial_loss_mode not in valid_loss_modes or args.regression_loss_mode not in valid_loss_modes:
        print(f"警告: 无效的损失函数模式。可选值: {valid_loss_modes}")
    if args.monitor_metric not in ['val_mse', 'val_acc', 'val_bce']:
        print(f"警告: 无效的监控指标。可选值: 'val_mse', 'val_acc', 'val_bce'")

    return args


# ==============================================================================


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    parser = setup_arg_parser()
    args = parser.parse_args()
    args = validate_args(args)

    print("===== 当前最终运行参数 (已合并默认值和命令行参数) =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("======================================================")

    stock_files_to_process = run_multi_gan.find_stock_files(args.data_base_dir, args.stock_name, args.sector)
    if not stock_files_to_process:
        print(f"错误: 在 '{args.data_base_dir}' 目录中找不到任何匹配的股票数据。请确认 get_stock_data.py 已成功运行。")
        sys.exit(1)

    print(f"\n成功找到 {len(stock_files_to_process)} 个股票数据文件待处理。")

    for stock_file in stock_files_to_process:
        path_parts = stock_file.replace('\\', '/').split('/')
        stock_name_from_path = path_parts[-2]
        sector_name_from_path = path_parts[-3]

        expected_output_dir = os.path.join(args.output_dir, sector_name_from_path, stock_name_from_path)

        if args.mode == "train" and os.path.isdir(expected_output_dir):
            ckpt_dir_check = os.path.join(expected_output_dir, 'ckpt', 'generators')
            if os.path.isdir(ckpt_dir_check) and any(f.endswith('.pt') for f in os.listdir(ckpt_dir_check)):
                print(f"\n--- 已找到股票 {stock_name_from_path} 的训练结果在 {expected_output_dir}，跳过训练。 ---")
                continue
            else:
                print(
                    f"\n--- 找到股票 {stock_name_from_path} 的目录 {expected_output_dir}，但未找到检查点，将尝试重新运行。 ---")

        try:
            run_multi_gan.run_experiment_for_stock(args, stock_file)

        except Exception as e:
            stock_name_from_path_in_err = stock_file.replace('\\', '/').split('/')[-2]
            print(f"\n!!!!!! 在处理股票 {stock_name_from_path_in_err} 时发生严重错误: {e} !!!!!!")
            import traceback
            traceback.print_exc()
            print(f"!!!!!! 跳过股票 {stock_name_from_path_in_err}，继续处理下一个... !!!!!!")
            continue

    print("\n所有指定的实验已全部完成。")


if __name__ == "__main__":
    main()