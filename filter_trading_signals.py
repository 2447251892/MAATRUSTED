# 文件名: filter_trading_signals.py

import pandas as pd
import os
import glob
import numpy as np
import traceback
import argparse
import sys

# --- 配置区 ---

# 信号生成模式:
# 1: 使用模型的分类预测 (predicted_action: 0, 1, 2)
# 2: 使用模型的股价预测 (下一日 predicted_close > 今日 predicted_close 来判断涨跌)
SIGNAL_GENERATION_MODE = 2

# 使用模式2时，如果下一日 predicted_close > 今日 predicted_close * (1 + PREDICTED_UP_THRESHOLD)，则视为买入信号
# 如果下一日 predicted_close < 今日 predicted_close * (1 - PREDICTED_DOWN_THRESHOLD)，则视为卖出信号
# 否则视为持平
PREDICTED_UP_THRESHOLD = 0.00
PREDICTED_DOWN_THRESHOLD = 0.00

# 过滤规则：买入后涨停时不卖出。定义涨停的百分比阈值。
LIMIT_UP_PCT_THRESHOLD = 9.5

# 原始信号文件目录
ORIGINAL_SIGNALS_BASE_DIR = 'output/multi_gan'
# 原始预测数据目录 (包含K线数据)
RAW_DATA_BASE_DIR = 'csv_data/predict'
# 过滤后信号文件保存目录
FILTERED_OUTPUT_BASE_DIR = 'output_filtered_signals'

# 初始本金（用于累计收益率计算）
INITIAL_CAPITAL = 1000000

# 手续费率 (万分之2.5，忽略最低5元限制)
# 0.025% = 0.00025
TRANSACTION_FEE_RATE = 0.00025


# --- 辅助函数 ---

def generate_initial_signals(df_merged: pd.DataFrame, mode: int, up_threshold: float,
                             down_threshold: float) -> pd.Series:
    if mode == 1:
        return df_merged['predicted_action']
    elif mode == 2:
        predicted_action_mode2 = pd.Series(1, index=df_merged.index, dtype=int)
        df_merged_sorted = df_merged.sort_values('date')
        prev_predicted_close = df_merged_sorted['predicted_close'].shift(1)
        buy_condition = df_merged_sorted['predicted_close'] > prev_predicted_close * (1 + up_threshold)
        sell_condition = df_merged_sorted['predicted_close'] < prev_predicted_close * (1 - down_threshold)
        predicted_action_mode2[buy_condition] = 2
        predicted_action_mode2[sell_condition] = 0
        predicted_action_mode2[prev_predicted_close.isna()] = 1
        return predicted_action_mode2.reindex(df_merged.index)
    else:
        raise ValueError(f"无效的信号生成模式: {mode}")


def apply_limit_up_filter(df_merged: pd.DataFrame, action_col_name: str, limit_up_threshold: float) -> pd.Series:
    df = df_merged.copy()
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    df.sort_values('date_dt', inplace=True)
    df['actual_prev_close'] = df['close'].shift(1)
    valid_prev_close_mask = (df['actual_prev_close'].notna()) & (df['actual_prev_close'] != 0)
    df['actual_pct_chg'] = np.nan
    df.loc[valid_prev_close_mask, 'actual_pct_chg'] = (df.loc[valid_prev_close_mask, 'close'] / df.loc[
        valid_prev_close_mask, 'actual_prev_close'] - 1) * 100
    limit_up_days_mask = (df['actual_pct_chg'].notna()) & (df['actual_pct_chg'] > limit_up_threshold)
    filtered_action = df[action_col_name].copy()
    sell_on_limit_up_mask = (filtered_action == 0) & limit_up_days_mask
    filtered_action[sell_on_limit_up_mask] = 1
    df.drop(columns=['date_dt', 'actual_prev_close', 'actual_pct_chg'], inplace=True, errors='ignore')
    return filtered_action.reindex(df_merged.index)


def calculate_trade_returns(df_merged: pd.DataFrame, action_col_name: str, fee_rate: float) -> pd.Series:
    df = df_merged.copy()
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    df.sort_values('date_dt', inplace=True)
    trade_return_pct = pd.Series(np.nan, index=df.index)
    holding = False
    buy_price = np.nan

    df['next_day_open'] = df['open'].shift(-1)

    for i in range(len(df)):
        current_action = df.iloc[i][action_col_name]

        if current_action == 2 and not holding:
            if pd.notna(df.iloc[i]['next_day_open']):
                holding = True
                buy_price = df.iloc[i]['next_day_open']

        elif current_action == 0 and holding:
            if pd.notna(df.iloc[i]['next_day_open']):
                sell_price = df.iloc[i]['next_day_open']
                if pd.notna(buy_price) and buy_price != 0:
                    # Calculate gross return multiplier
                    gross_multiplier = sell_price / buy_price

                    # Calculate net return percentage considering fees (ignoring min fee)
                    # Net return multiplier = (sell_price * (1 - fee_rate)) / (buy_price * (1 + fee_rate))
                    # simplified: Net return multiplier = (sell_price / buy_price) * ((1 - fee_rate) / (1 + fee_rate))
                    # However, fees are on transaction value.
                    # Percentage return formula used: (r * (1 - fee_rate) - (1 + fee_rate)) * 100 / (1 + fee_rate) is net gain / initial total cost
                    # Simpler: percentage return = ((sell_price * (1-fee_rate)) - (buy_price * (1+fee_rate))) / (buy_price * (1+fee_rate)) * 100
                    # Percentage return = ((sell_price * 0.99975 - buy_price * 1.00025) / (buy_price * 1.00025)) * 100

                    net_multiplier = (gross_multiplier * (
                                1 - fee_rate) - fee_rate)  # simplified net return multiplier relative to 1 unit of buy value

                    # Percentage return relative to buy price
                    # Net profit per share = sell_price * (1-fee_rate) - buy_price * (1+fee_rate)
                    # Percentage return = (Net profit per share / buy_price) * 100
                    net_profit_per_share = sell_price * (1 - fee_rate) - buy_price * (1 + fee_rate)
                    profit_pct = (net_profit_per_share / buy_price) * 100

                    trade_return_pct.loc[df.index[i]] = profit_pct
                holding = False
                buy_price = np.nan

    df.drop(columns=['date_dt', 'next_day_open'], inplace=True, errors='ignore')
    return trade_return_pct.reindex(df_merged.index)


def calculate_strategy_metrics(trade_returns_series: pd.Series, initial_capital: float) -> dict:
    completed_trades = trade_returns_series.dropna()
    num_trades = len(completed_trades)

    if num_trades == 0:
        return {
            'num_trades': 0,
            'cumulative_return_percentage': 0.0,
            'avg_return_per_trade_percentage': 0.0,
            'win_rate_percentage': 0.0,
            'profit_loss_ratio': np.nan  # Infinity or NaN if no losses
        }

    winning_trades = completed_trades[completed_trades > 0]
    losing_trades = completed_trades[completed_trades < 0]

    total_profit = winning_trades.sum()
    total_loss = losing_trades.sum()

    if total_loss == 0:
        profit_loss_ratio = np.inf if total_profit > 0 else np.nan
    else:
        profit_loss_ratio = abs(total_profit / total_loss)

    num_winning_trades = len(winning_trades)
    win_rate_percentage = (num_winning_trades / num_trades) * 100 if num_trades > 0 else 0.0

    avg_return_per_trade_percentage = completed_trades.mean()

    # Cumulative return (compounding)
    decimal_returns = completed_trades / 100.0
    # Assuming each trade is reinvested
    cumulative_multiplier = (1 + decimal_returns).prod()
    cumulative_return_percentage = (cumulative_multiplier - 1) * 100.0

    # If only one trade or calculation resulted in non-finite number
    if pd.isna(cumulative_return_percentage) or np.isinf(cumulative_return_percentage):
        cumulative_return_percentage = 0.0

    return {
        'num_trades': num_trades,
        'cumulative_return_percentage': cumulative_return_percentage,
        'avg_return_per_trade_percentage': avg_return_per_trade_percentage,
        'win_rate_percentage': win_rate_percentage,
        'profit_loss_ratio': profit_loss_ratio
    }


def process_single_signal_file(original_signal_filepath: str, original_signals_base_dir: str, raw_data_base_dir: str,
                               filtered_output_base_dir: str, mode: int, up_threshold: float, down_threshold: float,
                               limit_up_threshold: float, initial_capital: float, fee_rate: float):
    try:
        relative_path = os.path.relpath(original_signal_filepath, original_signals_base_dir)
        path_parts = relative_path.split(os.sep)
        if len(path_parts) < 2:
            print(f"跳过路径结构异常的文件: {original_signal_filepath}")
            return False
        sector_name = path_parts[0]
        stock_name = path_parts[1]
        signal_filename = path_parts[-1]
        print(f"\n正在处理 {sector_name}/{stock_name}/{signal_filename}... (模式: {mode})")

        raw_data_pattern = os.path.join(raw_data_base_dir, sector_name, stock_name, "*.csv")
        raw_data_files = glob.glob(raw_data_pattern)
        if not raw_data_files:
            print(f"  错误: 未找到原始数据文件 {sector_name}/{stock_name} 在 {raw_data_base_dir}。跳过。")
            return False
        raw_data_filepath = raw_data_files[0]

        signal_cols_to_load = ['date', 'predicted_action', 'predicted_close']
        df_signals = pd.read_csv(original_signal_filepath, dtype={'date': str},
                                 usecols=lambda c: c in signal_cols_to_load)

        if df_signals.empty:
            print(f"  警告: {stock_name} 的信号文件 {signal_filename} 为空。跳过。")
            return False

        if mode == 2 and 'predicted_close' not in df_signals.columns:
            print(f"  错误: 模式2需要 'predicted_close' 列，但信号文件 {signal_filename} 中未找到。跳过。")
            return False

        raw_cols_to_load = ['date', 'close', 'open']
        df_raw = pd.read_csv(raw_data_filepath, dtype={'date': str}, usecols=lambda c: c in raw_cols_to_load)

        df_signals['date_dt'] = pd.to_datetime(df_signals['date'], format='%Y%m%d', errors='coerce')
        df_raw['date_dt'] = pd.to_datetime(df_raw['date'], format='%Y%m%d', errors='coerce')

        df_signals.dropna(subset=['date_dt'], inplace=True)
        df_raw.dropna(subset=['date_dt'], inplace=True)

        df_signals.sort_values('date_dt', inplace=True)
        df_raw.sort_values('date_dt', inplace=True)

        df_merged = pd.merge(df_raw[['date_dt', 'close', 'open']],
                             df_signals[['date_dt', 'predicted_action', 'predicted_close']], on='date_dt', how='inner')
        df_merged.rename(columns={'date_dt': 'date'}, inplace=True)

        if df_merged.empty:
            print(f"  警告: 合并数据后为空。跳过。")
            return False

        df_merged = df_merged.reset_index(drop=True)

        df_merged['predicted_action_initial'] = generate_initial_signals(
            df_merged, mode, up_threshold, down_threshold
        )

        df_merged['predicted_action_filtered'] = apply_limit_up_filter(
            df_merged, 'predicted_action_initial', limit_up_threshold=limit_up_threshold
        )

        df_merged['trade_return_pct'] = calculate_trade_returns(df_merged, 'predicted_action_filtered',
                                                                fee_rate=fee_rate)

        strategy_metrics = calculate_strategy_metrics(df_merged['trade_return_pct'], initial_capital=initial_capital)

        df_final_signals = df_merged[
            ['date', 'predicted_action_filtered', 'predicted_close', 'trade_return_pct']].copy()
        df_final_signals.rename(columns={'predicted_action_filtered': 'predicted_action'}, inplace=True)

        output_dir = os.path.join(filtered_output_base_dir, sector_name, stock_name)
        os.makedirs(output_dir, exist_ok=True)

        filtered_signal_filepath = os.path.join(output_dir, signal_filename)
        metrics_filepath = os.path.join(output_dir, signal_filename.replace('_daily_signals.csv', '_metrics.csv'))

        df_final_signals.to_csv(filtered_signal_filepath, index=False, float_format='%.4f')

        df_metrics = pd.DataFrame([strategy_metrics])
        df_metrics.to_csv(metrics_filepath, index=False, float_format='%.4f')

        print(f"  成功过滤、计算收益和指标并保存到:\n    信号: {filtered_signal_filepath}\n    指标: {metrics_filepath}")

        return True

    except Exception as e:
        print(f"\n  处理文件 {original_signal_filepath} 时发生错误: {e}\n{traceback.format_exc()}")
        return False


# --- 主执行 ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="根据特定规则过滤交易信号并计算收益和指标。")
    parser.add_argument('--input_dir', type=str, default=ORIGINAL_SIGNALS_BASE_DIR,
                        help=f'包含原始信号文件的基础目录 (例如 {ORIGINAL_SIGNALS_BASE_DIR})。')
    parser.add_argument('--raw_data_dir', type=str, default=RAW_DATA_BASE_DIR,
                        help=f'包含原始预测数据文件的基础目录 (例如 {RAW_DATA_BASE_DIR})。')
    parser.add_argument('--output_dir', type=str, default=FILTERED_OUTPUT_BASE_DIR,
                        help=f'保存过滤后信号文件和收益的的基础目录 (例如 {FILTERED_OUTPUT_BASE_DIR})。')
    # 可以根据需要添加命令行参数来覆盖配置文件中的模式和阈值
    # parser.add_argument('--mode', type=int, help='信号生成模式 (1或2)')
    # parser.add_argument('--up_thresh', type=float, help='模式2上涨阈值')
    # parser.add_argument('--down_thresh', type=float, help='模式2下跌阈值')
    # parser.add_argument('--limit_up_thresh', type=float, help='涨停过滤阈值')
    # parser.add_argument('--initial_capital', type=float, help='初始本金')
    # parser.add_argument('--fee_rate', type=float, help='单边手续费率 (例如 0.00025)')

    args = parser.parse_args()

    signal_mode = args.mode if hasattr(args, 'mode') and args.mode is not None else SIGNAL_GENERATION_MODE
    up_thresh = args.up_thresh if hasattr(args, 'up_thresh') and args.up_thresh is not None else PREDICTED_UP_THRESHOLD
    down_thresh = args.down_thresh if hasattr(args,
                                              'down_thresh') and args.down_thresh is not None else PREDICTED_DOWN_THRESHOLD
    limit_up_thresh = args.limit_up_thresh if hasattr(args,
                                                      'limit_up_thresh') and args.limit_up_thresh is not None else LIMIT_UP_PCT_THRESHOLD
    initial_capital = args.initial_capital if hasattr(args,
                                                      'initial_capital') and args.initial_capital is not None else INITIAL_CAPITAL
    fee_rate = args.fee_rate if hasattr(args, 'fee_rate') and args.fee_rate is not None else TRANSACTION_FEE_RATE

    print(f"正在 {args.input_dir} 中扫描原始信号文件...")
    original_signal_files_pattern = os.path.join(args.input_dir, '**', '*_daily_signals.csv')
    original_signal_filepaths = glob.glob(original_signal_files_pattern, recursive=True)

    if not original_signal_filepaths:
        print(f"未找到符合模式的原始信号文件: {original_signal_files_pattern}")
        sys.exit(0)

    print(f"找到 {len(original_signal_filepaths)} 个原始信号文件待处理。")
    print(f"当前信号生成模式: {signal_mode}")
    if signal_mode == 2:
        print(f"  模式2阈值: 上涨 > {up_thresh * 100:.2f}%, 下跌 < -{down_thresh * 100:.2f}%")
    print(f"涨停过滤阈值: > {limit_up_thresh:.2f}%")
    print(f"初始本金: {initial_capital:.2f}")
    print(f"手续费率: 万分之 {fee_rate * 10000:.2f} (单边，忽略最低5元)")

    processed_count = 0
    failed_files = []

    for filepath in original_signal_filepaths:
        success = process_single_signal_file(
            filepath, args.input_dir, args.raw_data_dir, args.output_dir,
            signal_mode, up_thresh, down_thresh, limit_up_thresh, initial_capital, fee_rate
        )
        if success:
            processed_count += 1
        else:
            failed_files.append(filepath)

    print("\n--- 过滤和收益计算总结 ---")
    print(f"尝试处理: {len(original_signal_filepaths)} 个文件")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {len(failed_files)} 个文件")
    if failed_files:
        print("失败文件列表:")
        for f in failed_files:
            print(f" - {f}")

    print("脚本执行完毕。")