# 文件名: experiment_runner.py

import argparse
import torch
import os
import sys
import time
import pandas as pd
# 导入功能模块
from run_multi_gan import find_stock_files, run_experiment_for_stock

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
    "num_epochs": 100,
    "lr": 2e-5,
    "batch_size": 64,
    "train_split": 0.8,
    "random_seed": 3407,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp_dtype": "none",
    "mode": "train",
    "ckpt_path": "auto",
}


# ==============================================================================

def setup_arg_parser():
    """设置命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="为预处理好的股票数据运行Multi-GAN模型。这是实验的主控制脚本。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for key, value in COMMON_ARGS.items():
        if isinstance(value, list):
            parser.add_argument(f'--{key}', type=type(value[0]) if value else str, nargs='+', default=value,
                                help=f"默认为: {value}")
        elif isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store_true', default=value, help=f"设置此项以激活 {key}")
        else:
            parser.add_argument(f'--{key}', type=type(value) if value is not None else str, default=value,
                                help=f"默认为: {value}")
    return parser


def validate_args(args):
    """验证和修正参数的逻辑。"""
    if len(args.generators) != args.N_pairs or len(args.window_sizes) != args.N_pairs:
        print("错误: --generators, --window_sizes, 和 --N_pairs 的数量必须一致!")
        sys.exit(1)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"警告: 指定了设备 {args.device} 但CUDA不可用，将使用CPU。")
        args.device = "cpu"
    return args


def main():
    """实验的主函数，负责解析参数和调度实验。"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    args = validate_args(args)

    print("===== 当前最终运行参数 (已合并默认值和命令行参数) =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("======================================================")

    stock_files_to_process = find_stock_files(args.data_base_dir, args.stock_name, args.sector)
    if not stock_files_to_process:
        print(f"错误: 在 '{args.data_base_dir}' 目录中找不到任何匹配的股票数据。请确认 get_stock_data.py 已成功运行。")
        sys.exit(1)

    print(f"\n成功找到 {len(stock_files_to_process)} 个股票数据文件待处理。")

    for stock_file in stock_files_to_process:
        try:
            run_experiment_for_stock(args, stock_file)
        except Exception as e:
            stock_name_from_path = stock_file.replace('\\', '/').split('/')[-2]
            print(f"\n!!!!!! 在处理股票 {stock_name_from_path} 时发生严重错误: {e} !!!!!!")
            import traceback
            traceback.print_exc()
            print(f"!!!!!! 跳过股票 {stock_name_from_path}，继续处理下一个... !!!!!!")
            continue

    print("\n所有指定的实验已全部完成。")


if __name__ == "__main__":
    main()