# 文件名: filter_trading_signals.py

import pandas as pd
import os
import glob
import numpy as np
import traceback
import argparse
import sys
import math
from datetime import timedelta
import multiprocessing
import shutil
# --- 新增绘图相关的库 ---
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.ticker as mticker


# ==============================================================================
# --- 全局字体配置变量 ---
# ==============================================================================
def setup_matplotlib_font():
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK JP', 'Source Han Sans CN',
                     'Arial Unicode MS']
    found_font = next((font_name for font_name in chinese_fonts if
                       any(font_name.lower() in font.name.lower() for font in
                           matplotlib.font_manager.fontManager.ttflist)),
                      None)
    if found_font:
        plt.rcParams['font.family'] = [found_font, 'sans-serif']
        return f"绘图模块已加载中文字体: {found_font}"
    else:
        plt.rcParams['font.family'] = ['sans-serif']
        return "警告: 未找到指定的中文字体, 图表中的中文可能无法正常显示。"


FONT_SETUP_MESSAGE = setup_matplotlib_font()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# ==============================================================================
# --- 总配置区 ---
# ==============================================================================
INITIAL_CAPITAL = 1000000
TRANSACTION_FEE_RATE = 0.00025
MIN_TRANSACTION_FEE = 5.0
ORIGINAL_SIGNALS_BASE_DIR = 'output/multi_gan'
RAW_DATA_BASE_DIR = 'csv_data/predict'
FILTERED_OUTPUT_BASE_DIR = 'output_filtered_signals'

# ==================== 策略开关配置 (布尔值: True/False) ====================
# 这里定义了各种交易策略的开关。True表示启用该策略，False表示禁用。
# 这些开关共同决定了在回测中如何过滤原始的买卖信号。
STRATEGY_CONFIG = {
    # 卖出过滤策略
    "LIMIT_UP_NO_SELL": True,  # 当日涨停时，即使有卖出信号也强制持有不卖出
    "ONE_WORD_BOARD_SELL": True,  # 启用“一字板”次日卖出策略（检测到昨日一字板涨停，今日放量则卖出）

    # 止损/止盈类策略
    "STOP_LOSS": False,  # 启用基于买入成本的固定比例止损

    # 持有条件策略 (决定是否继续持有股票)
    "CONSECUTIVE_UP_HOLD": False,  # (未实现) 连续上涨N天后，即使有卖出信号也继续持有
    "HOLD_ABOVE_MA": False,  # (未实现) 只要股价在某条移动平均线之上，就无视卖出信号继续持有
    "NO_DROP_SINCE_BUY_HOLD": False,  # (未实现) 自买入以来，如果股价从未跌破买入价，则无视卖出信号继续持有

    # 买入过滤策略
    "DAILY_DROP_LIMIT": True,  # 当日跌停（或接近跌停）时，即使有买入信号也禁止买入
    "ONE_WORD_BOARD_NO_BUY": True,  # 当日一字板涨停时，即使有买入信号也禁止买入
    "LOSS_BAN": False,  # 在一笔亏损交易后，禁止在N天内再次买入
    "DAILY_GAIN_LIMIT": False,  # 当日涨幅过高（如超过6%）时，即使有买入信号也禁止买入
    "MA_TREND": False,  # 仅当短期均线在长期均线之上（多头趋势）时才允许买入
    "RSI_CHECK": False,  # (未实现) 仅当RSI指标未处于超买区时才允许买入
    "ADX_CHECK": False,  # 仅当ADX指标显示有趋势时才允许买入
    "PREDICTED_PRICE_INCREASE": False,  # (未实现) 仅当模型预测的下一个价格高于当前价格时才买入
    "GAP_UP_FALL_BACK": False,  # (未实现) 当日大幅高开后回落时，即使有买入信号也禁止买入
}

# ==================== 策略参数配置 (数值) ====================
# 这里定义了与上述策略开关相对应的具体数值参数。
STRATEGY_PARAMS = {
    # 模式2信号生成参数
    "PREDICTED_UP_THRESHOLD": 0.00,  # 在模式2中，预测价格需比前一日高出此百分比才生成买入信号(0.01 = 1%)
    "PREDICTED_DOWN_THRESHOLD": 0.00,  # 在模式2中，预测价格需比前一日低出此百分比才生成卖出信号(0.01 = 1%)

    # 止损/止盈参数
    "STOP_LOSS_THRESHOLD": -0.2,  # 止损阈值，当亏损达到20%时触发 (-0.2 = -20%)
    "LIMIT_UP_THRESHOLD": 9.6,  # 定义“涨停”的涨幅阈值 (9.6%)

    # 卖出条件参数
    "SELL_SMA_PERIOD": 5,  # 用于“跌破均线卖出”策略的移动平均线周期

    # “一字板”策略相关参数
    "ONE_WORD_BOARD_PCT_CHG_THRESHOLD": 9.6,  # 定义“一字板”的涨幅阈值
    "ONE_WORD_BOARD_VOLUME_MULTIPLIER": 4.0,  # “一字板”次日成交量需要是前一日的多少倍才触发卖出
    "ONE_WORD_BOARD_PRICE_DIFF_TOLERANCE": 0.01,  # 定义“一字板”时，开盘价与收盘价差异的容忍度 (1%)
    "ONE_WORD_BOARD_SELL_BAN_DAYS": 10,  # 在执行“一字板”卖出策略后，禁止买入该股票的天数

    # 买入条件参数
    "LOSS_BAN_DAYS": 3,  # 亏损交易后，禁止买入的天数
    "DAILY_GAIN_LIMIT": 6.0,  # 定义“涨幅过高”的阈值 (6.0%)
    "DAILY_DROP_LIMIT": -9.5,  # 定义“跌停”的跌幅阈值 (-9.5%)
    # 均线趋势判断参数
    "SHORT_MA_PERIOD_FOR_TREND": 5,
    "LONG_MA_PERIOD_FOR_TREND": 20,
    "MA_PERIOD_FOR_TREND": 5,
    "GAP_UP_THRESHOLD": 8.0,  # 定义“大幅高开”的阈值 (8.0%)

    # 技术指标相关阈值
    "RSI_OVERSOLD_THRESHOLD": 80,  # RSI超买区的阈值
    "ADX_BUY_THRESHOLD": 20,  # ADX趋势强度阈值
}


# ==============================================================================
# --- 辅助函数 ---
# ==============================================================================
def plot_trade_analysis(trades_df, daily_equity_curve, stock_info, output_dir):
    stock_name = stock_info['stock_name']
    generator_index = stock_info['generator_index']
    mode = stock_info['mode']

    os.makedirs(output_dir, exist_ok=True)

    if not trades_df.empty:
        plt.figure(figsize=(14, 7))
        colors = ['red' if r > 0 else 'green' for r in trades_df['return']]
        plt.bar(range(len(trades_df)), trades_df['return'] * 100, color=colors)
        plt.title(f'{stock_name} (G{generator_index}, Mode {mode}) - 每笔交易收益率', fontsize=16)
        plt.xlabel('交易笔数', fontsize=12)
        plt.ylabel('收益率 (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(0, color='gray', linewidth=0.8)
        plt.tight_layout(pad=1.5)
        plt.savefig(os.path.join(output_dir, f'G{generator_index}_mode{mode}_individual_trades_returns.png'), dpi=300)
        plt.close()

    if not daily_equity_curve.empty:
        fig, (ax_equity, ax_drawdown) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                                     gridspec_kw={'height_ratios': [3, 1]})

        ax_equity.plot(daily_equity_curve.index, daily_equity_curve, label='策略累积净值', color='blue', linewidth=2)
        ax_equity.set_title(f'{stock_name} (G{generator_index}, Mode {mode}) - 累积收益与回撤', fontsize=16)
        ax_equity.set_ylabel('组合净值', fontsize=12)
        ax_equity.grid(True, linestyle='--', alpha=0.6)
        ax_equity.legend(loc='upper left')

        rolling_max = daily_equity_curve.expanding().max()
        drawdown = (rolling_max - daily_equity_curve) / rolling_max
        ax_drawdown.fill_between(daily_equity_curve.index, 0, drawdown, color='green', alpha=0.3, label='回撤')
        ax_drawdown.set_ylabel('回撤', fontsize=12)
        ax_drawdown.set_xlabel('日期', fontsize=12)
        ax_drawdown.grid(True, linestyle='--', alpha=0.6)
        ax_drawdown.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax_drawdown.set_ylim(bottom=0)

        plt.tight_layout(pad=1.5)
        plt.savefig(os.path.join(output_dir, f'G{generator_index}_mode{mode}_cumulative_return_drawdown.png'), dpi=300)
        plt.close()


def generate_initial_signals(df_merged: pd.DataFrame, mode: int) -> pd.Series:
    if 'predicted_action_from_model' not in df_merged.columns: raise KeyError("'predicted_action'列缺失。")
    if mode == 1:
        return df_merged['predicted_action_from_model']
    elif mode == 2:
        if 'predicted_close' not in df_merged.columns: raise KeyError("模式2需要'predicted_close'列。")
        actions = pd.Series(1, index=df_merged.index, dtype=int);
        df_s = df_merged.sort_values('date')
        prev_close = df_s['predicted_close'].shift(1)
        actions[df_s['predicted_close'] > prev_close * (1 + STRATEGY_PARAMS["PREDICTED_UP_THRESHOLD"])] = 2
        actions[df_s['predicted_close'] < prev_close * (1 - STRATEGY_PARAMS["PREDICTED_DOWN_THRESHOLD"])] = 0
        actions[prev_close.isna()] = 1
        return actions.reindex(df_merged.index)
    raise ValueError(f"无效模式: {mode}")


# ==============================================================================
# --- 主处理函数 (核心回测逻辑) ---
# ==============================================================================
def run_backtest_for_strategy(task_info):
    original_signal_filepath, signal_gen_mode = task_info['filepath'], task_info['mode']
    try:
        relative_path = os.path.relpath(original_signal_filepath, ORIGINAL_SIGNALS_BASE_DIR)
        path_parts = relative_path.replace('\\', '/').split('/')
        if len(path_parts) < 3: return None
        sector_name, stock_name, original_filename = path_parts[0], path_parts[1], path_parts[2]

        try:
            gen_idx_str = original_filename.split('_')[0].replace('G', '')
            generator_index = int(gen_idx_str)
        except (ValueError, IndexError):
            return None

        ts_code = task_info.get('ts_code', 'N/A')
        print(f"正在处理 {stock_name}({ts_code}) G{generator_index} (模式 {signal_gen_mode})...")

        raw_data_pattern = os.path.join(RAW_DATA_BASE_DIR, sector_name, stock_name, "*.csv")
        raw_data_files = glob.glob(raw_data_pattern)
        if not raw_data_files: return None

        short_ma_period = STRATEGY_PARAMS.get("SHORT_MA_PERIOD_FOR_TREND", 5)
        long_ma_period = STRATEGY_PARAMS.get("LONG_MA_PERIOD_FOR_TREND", 20)
        adx_period = 14

        required_cols_raw = ['date', 'open', 'high', 'low', 'close', 'volume', 'pctChg']

        if STRATEGY_CONFIG.get("MA_TREND", False):
             required_cols_raw.extend([f'sma_{short_ma_period}', f'sma_{long_ma_period}'])
        if STRATEGY_CONFIG.get("ADX_CHECK", False):
             required_cols_raw.append(f'adx_{adx_period}')


        required_cols_raw = list(set(required_cols_raw))

        df_raw = pd.read_csv(raw_data_files[0], dtype={'date': str}, usecols=lambda c: c in required_cols_raw)
        df_signals = pd.read_csv(original_signal_filepath, dtype={'date': str})

        df_raw['date'] = pd.to_datetime(df_raw['date'], format='%Y%m%d', errors='coerce')
        df_signals['date'] = pd.to_datetime(df_signals['date'], format='%Y%m%d', errors='coerce')
        df_signals.rename(columns={'predicted_action': 'predicted_action_from_model'}, inplace=True)
        df_merged = pd.merge(df_raw, df_signals, on='date', how='inner').sort_values('date').set_index('date')
        if df_merged.empty: return None

        df_merged['predicted_action'] = generate_initial_signals(df_merged.reset_index(), signal_gen_mode).values

        df_merged['open_T+1'] = df_merged['open'].shift(-1)
        df_merged['open_prev1'] = df_merged['open'].shift(1)
        df_merged['close_prev1'] = df_merged['close'].shift(1)
        df_merged['pctChg_prev1'] = df_merged['pctChg'].shift(1)
        df_merged['volume_prev1'] = df_merged['volume'].shift(1)

        trades_log = [];
        daily_actions_log = [];
        current_position = None
        shares_bought, total_buy_cost = 0, 0.0
        one_word_board_ban_end_date = pd.NaT
        loss_ban_end_date = pd.NaT

        daily_equity = 1.0
        equity_curve_points = []
        if not df_merged.empty:
            equity_curve_points.append((df_merged.index[0], 1.0))

        trading_dates = df_merged.index

        for date_T, row_T in df_merged.iterrows():
            pred_for_Tplus1 = row_T['predicted_action']
            action_to_take = None

            current_loc = trading_dates.get_loc(date_T)
            is_not_last_day = current_loc + 1 < len(trading_dates)
            next_trade_date = trading_dates[current_loc + 1] if is_not_last_day else pd.NaT


            if current_position == 'long':
                if STRATEGY_CONFIG["ONE_WORD_BOARD_SELL"]:
                    open_d_m1 = row_T.get('open_prev1');
                    close_d_m1 = row_T.get('close_prev1');
                    pct_d_m1 = row_T.get('pctChg_prev1');
                    vol_d_m1 = row_T.get('volume_prev1')
                    if all(pd.notna(x) for x in [open_d_m1, close_d_m1, pct_d_m1, vol_d_m1, row_T['volume']]):
                        is_dm1_one_word_board = (
                                pct_d_m1 >= STRATEGY_PARAMS["ONE_WORD_BOARD_PCT_CHG_THRESHOLD"] and abs(
                            open_d_m1 - close_d_m1) / open_d_m1 < STRATEGY_PARAMS[
                                    "ONE_WORD_BOARD_PRICE_DIFF_TOLERANCE"])
                        is_vol_spike = (vol_d_m1 > 0 and row_T['volume'] >= vol_d_m1 * STRATEGY_PARAMS[
                            "ONE_WORD_BOARD_VOLUME_MULTIPLIER"])
                        if is_dm1_one_word_board and is_vol_spike:
                            action_to_take = 'Sell_One_Word'


                if action_to_take is None:
                    is_hold_filtered = False
                    if STRATEGY_CONFIG["LIMIT_UP_NO_SELL"] and row_T['pctChg'] >= STRATEGY_PARAMS["LIMIT_UP_THRESHOLD"]:
                        is_hold_filtered = True

                    if not is_hold_filtered and pred_for_Tplus1 == 0:
                        action_to_take = 'Sell'


            elif current_position is None:
                can_buy = (pred_for_Tplus1 == 2)

                if can_buy and pd.notna(next_trade_date) and next_trade_date < one_word_board_ban_end_date: can_buy = False
                if can_buy and STRATEGY_CONFIG["DAILY_DROP_LIMIT"] and row_T['pctChg'] <= STRATEGY_PARAMS["DAILY_DROP_LIMIT"]: can_buy = False

                if can_buy and STRATEGY_CONFIG.get("ONE_WORD_BOARD_NO_BUY", False):
                    is_today_one_word_board = False
                    if row_T['open'] > 0 and row_T['close'] > 0 and row_T['volume'] > 0:
                         is_today_one_word_board = (
                                row_T['pctChg'] >= STRATEGY_PARAMS["ONE_WORD_BOARD_PCT_CHG_THRESHOLD"] and
                                abs(row_T['open'] - row_T['close']) / row_T['open'] < STRATEGY_PARAMS[
                                    "ONE_WORD_BOARD_PRICE_DIFF_TOLERANCE"]
                        )
                    if is_today_one_word_board:
                        can_buy = False

                if can_buy and STRATEGY_CONFIG.get("LOSS_BAN", False) and pd.notna(next_trade_date) and next_trade_date < loss_ban_end_date:
                     can_buy = False

                if can_buy and STRATEGY_CONFIG.get("MA_TREND", False):
                    short_ma_col = f'sma_{short_ma_period}'
                    long_ma_col = f'sma_{long_ma_period}'
                    if short_ma_col in row_T and long_ma_col in row_T:
                        short_ma_val = row_T[short_ma_col]
                        long_ma_val = row_T[long_ma_col]
                        if pd.isna(short_ma_val) or pd.isna(long_ma_val) or short_ma_val <= long_ma_val:
                            can_buy = False
                    else:
                         can_buy = False

                if can_buy and STRATEGY_CONFIG.get("ADX_CHECK", False):
                    adx_col = f'adx_{adx_period}'
                    if adx_col in row_T:
                         adx_val = row_T[adx_col]
                         if pd.isna(adx_val) or adx_val < STRATEGY_PARAMS["ADX_BUY_THRESHOLD"]:
                             can_buy = False
                    else:
                         can_buy = False


                if can_buy:
                    action_to_take = "Buy"

            filtered_action_str = "Hold Long" if current_position == 'long' else "Hold Cash"
            if action_to_take == 'Buy': filtered_action_str = 'Buy'
            if action_to_take in ['Sell', 'Sell_One_Word']: filtered_action_str = 'Sell'

            daily_actions_log.append(
                {'date': date_T, 'predicted_action': pred_for_Tplus1, 'filtered_action': filtered_action_str})

            if is_not_last_day:
                if action_to_take in ["Sell_One_Word", "Sell"]:
                    sell_price = row_T['open_T+1']
                    if pd.notna(sell_price) and sell_price > 0 and current_position == 'long':
                        sell_value = shares_bought * sell_price;
                        sell_fee = max(sell_value * TRANSACTION_FEE_RATE, MIN_TRANSACTION_FEE)
                        net_proceeds = sell_value - sell_fee
                        trade_return = (net_proceeds - total_buy_cost) / total_buy_cost if total_buy_cost > 0 else 0.0
                        trades_log.append({'exit_date': next_trade_date, 'return': trade_return})
                        current_position = None
                        if action_to_take == "Sell_One_Word":
                            one_word_board_ban_end_date = next_trade_date + timedelta(days=STRATEGY_PARAMS["ONE_WORD_BOARD_SELL_BAN_DAYS"])
                        if STRATEGY_CONFIG.get("LOSS_BAN", False) and trade_return < 0:
                             loss_ban_end_date = next_trade_date + timedelta(days=STRATEGY_PARAMS["LOSS_BAN_DAYS"])

                        daily_equity *= (1 + trade_return)
                        equity_curve_points.append((next_trade_date, daily_equity))


                elif action_to_take == "Buy":
                    buy_price = row_T['open_T+1']
                    if pd.notna(buy_price) and buy_price > 0 and current_position is None:
                         can_execute_buy = True
                         if pd.notna(next_trade_date) and next_trade_date < one_word_board_ban_end_date: can_execute_buy = False
                         if pd.notna(next_trade_date) and next_trade_date < loss_ban_end_date: can_execute_buy = False

                         if can_execute_buy and (daily_equity * INITIAL_CAPITAL) > (buy_price * 100 + max(buy_price * 100 * TRANSACTION_FEE_RATE, MIN_TRANSACTION_FEE)):
                              shares_to_buy = math.floor((daily_equity * INITIAL_CAPITAL) / buy_price / 100) * 100
                              if shares_to_buy > 0:
                                  buy_value = shares_to_buy * buy_price;
                                  buy_fee = max(buy_value * TRANSACTION_FEE_RATE, MIN_TRANSACTION_FEE)
                                  current_position = 'long'
                                  total_buy_cost, shares_bought = buy_value + buy_fee, shares_to_buy


        if not equity_curve_points:
            daily_equity_curve = pd.Series(1.0, index=df_merged.index)
        else:
            equity_series = pd.Series([val for date, val in equity_curve_points],
                                      index=[date for date, val in equity_curve_points])
            daily_equity_curve = equity_series.reindex(df_merged.index, method='ffill')
            if not daily_equity_curve.empty and pd.isna(daily_equity_curve.iloc[0]): daily_equity_curve.iloc[0] = 1.0
            daily_equity_curve = daily_equity_curve.ffill()

        df_trades = pd.DataFrame(trades_log)
        num_trades = len(df_trades)
        if num_trades == 0:
            metrics = {'num_trades': 0, 'cumulative_return_percentage': 0.0, 'avg_return_per_trade_percentage': 0.0,
                       'win_rate_percentage': 0.0, 'profit_loss_ratio': np.nan, 'max_single_trade_drawdown_percentage': 0.0}
        else:
            win_rate = (df_trades['return'] > 0).sum() / num_trades * 100
            avg_return = df_trades['return'].mean() * 100
            total_profit = df_trades[df_trades['return'] > 0]['return'].sum()
            total_loss = abs(df_trades[df_trades['return'] < 0]['return'].sum())
            profit_loss_ratio = total_profit / total_loss if total_loss > 0 else np.inf
            cumulative_return = (daily_equity_curve.iloc[-1] - daily_equity_curve.iloc[0]) if not daily_equity_curve.empty else 0.0
            cumulative_return_pct = cumulative_return / daily_equity_curve.iloc[0] * 100 if not daily_equity_curve.empty and daily_equity_curve.iloc[0] != 0 else 0.0

            losing_returns = df_trades[df_trades['return'] < 0]['return']
            # Calculate max single trade drawdown as a negative percentage
            max_single_trade_drawdown_pct = 0.0
            if not losing_returns.empty:
                 max_single_trade_drawdown_pct = losing_returns.min() * 100 # min() gives the largest negative return

            metrics = {'num_trades': num_trades, 'cumulative_return_percentage': cumulative_return_pct,
                       'avg_return_per_trade_percentage': avg_return, 'win_rate_percentage': win_rate,
                       'profit_loss_ratio': profit_loss_ratio, 'max_single_trade_drawdown_percentage': max_single_trade_drawdown_pct}


        output_dir = os.path.join(FILTERED_OUTPUT_BASE_DIR, sector_name, stock_name)
        os.makedirs(output_dir, exist_ok=True)

        new_signal_filename = f'G{generator_index}_mode{signal_gen_mode}_daily_signals.csv'
        new_metrics_filename = f'G{generator_index}_mode{signal_gen_mode}_metrics.csv'

        df_filtered_signals = pd.DataFrame(daily_actions_log)
        if not df_trades.empty:
            df_trades_for_merge = df_trades[['exit_date', 'return']].rename(
                columns={'exit_date': 'date', 'return': 'trade_return_pct'})
            df_trades_for_merge['trade_return_pct'] *= 100
            df_filtered_signals = pd.merge(df_filtered_signals, df_trades_for_merge, on='date', how='left')

        df_filtered_signals['date'] = df_filtered_signals['date'].dt.strftime('%Y%m%d')
        df_filtered_signals.to_csv(os.path.join(output_dir, new_signal_filename), index=False, float_format='%.2f')
        pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, new_metrics_filename), index=False, float_format='%.4f')

        stock_info_for_plot = {'ts_code': ts_code, 'stock_name': stock_name, 'generator_index': generator_index,
                               'mode': signal_gen_mode}
        plot_trade_analysis(df_trades, daily_equity_curve, stock_info_for_plot, output_dir)

        return {'ts_code': ts_code, 'stock_name': stock_name, 'sector': sector_name, 'generator_index': generator_index,
                'mode': signal_gen_mode, **metrics}

    except Exception:
        traceback.print_exc()
        return None


# ==============================================================================
# --- 主执行 ---
# ==============================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    print(FONT_SETUP_MESSAGE)

    parser = argparse.ArgumentParser(description="自动测试所有模型的所有信号模式，并识别最佳策略。")
    parser.add_argument('--num_cores', type=int, default=multiprocessing.cpu_count(), help=f"并行核心数。")
    args = parser.parse_args()

    print("=" * 50 + "\n阶段 1: 扫描并生成所有待处理的回测任务\n" + "=" * 50)
    tasks = []
    signal_files = glob.glob(os.path.join(ORIGINAL_SIGNALS_BASE_DIR, '**', '*_daily_signals.csv'), recursive=True)
    if not signal_files:
        print(f"未找到任何原始信号文件: {os.path.join(ORIGINAL_SIGNALS_BASE_DIR, '**', '*_daily_signals.csv')}")
        sys.exit(0)
    for filepath in signal_files:
        try:
            norm_path = os.path.normpath(filepath)
            path_parts = norm_path.split(os.sep)
            if len(path_parts) >= 4:
                stock_name = path_parts[-2]
                sector_name = path_parts[-3]
                raw_data_dir = os.path.join(RAW_DATA_BASE_DIR, sector_name, stock_name)
                predict_files = glob.glob(os.path.join(raw_data_dir, '*_predict_data.csv'))
                if predict_files:
                    raw_csv_name = os.path.basename(predict_files[0])
                    ts_code_parts = raw_csv_name.split('_')
                    ts_code = f"{ts_code_parts[0]}.{ts_code_parts[1]}"
                    for mode in [1, 2]: tasks.append({'filepath': filepath, 'ts_code': ts_code, 'mode': mode})
        except(FileNotFoundError, IndexError):
            continue
    print(f"找到 {len(tasks)} 个策略-模式组合待回测。")

    print("=" * 50 + "\n阶段 2: 开始并行回测所有策略\n" + "=" * 50)
    if not tasks: sys.exit(0)
    num_cores = min(args.num_cores, len(tasks))
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(pool.map(run_backtest_for_strategy, tasks))
    successful_results = [r for r in results if r is not None]
    print(f"回测完成。成功数: {len(successful_results)}, 失败数: {len(tasks) - len(successful_results)}")

    print("=" * 50 + "\n阶段 3: 开始识别每支股票的最佳策略\n" + "=" * 50)
    if not successful_results:
        print("没有任何成功的回测结果，无法识别最佳策略。")
    else:
        df_results = pd.DataFrame(successful_results)
        unique_stocks = df_results.drop_duplicates(subset=['ts_code'])
        for _, stock_info in unique_stocks.iterrows():
            ts_code, sector, stock_name = stock_info['ts_code'], stock_info['sector'], stock_info['stock_name']
            print(f"--- 识别 {stock_name} ({ts_code}) 的最佳策略 ---")
            stock_metrics = df_results[df_results['ts_code'] == ts_code].to_dict('records')
            if not stock_metrics: continue


            def calculate_score(metric):
                if metric.get('num_trades', 0) <= 10: return -float('inf')
                cum_ret = metric.get('cumulative_return_percentage', 0) / 100.0;
                win_rate = metric.get('win_rate_percentage', 0) / 100.0
                pl_ratio = metric.get('profit_loss_ratio', 0.0)
                # Get the magnitude of the max single trade drawdown for the penalty
                max_dd_magnitude = abs(metric.get('max_single_trade_drawdown_percentage', 0.0)) / 100.0

                if cum_ret < 0: return -float('inf')
                log_pl = np.log1p(pl_ratio) if pl_ratio != float('inf') and pl_ratio is not None else np.log(1000)

                score = cum_ret * (win_rate ** 1.5) * log_pl * np.log1p(metric['num_trades'])

                if max_dd_magnitude > 0:
                    score = score / (1 + max_dd_magnitude * 5)

                return score if np.isfinite(score) else -float('inf')


            scored_metrics = sorted([m for m in stock_metrics if m.get('num_trades', 0) > 10], key=calculate_score,
                                    reverse=True)
            if scored_metrics and calculate_score(scored_metrics[0]) > -float('inf'):
                best_metric = scored_metrics[0]
            else:
                best_metric_fallback = None
                if stock_metrics:
                     best_metric_fallback = max(stock_metrics, key=lambda x: (x.get('num_trades', 0), x.get('cumulative_return_percentage', -float('inf'))))
                best_metric = best_metric_fallback if best_metric_fallback else {'generator_index': 'N/A', 'mode': 'N/A', 'num_trades': 0, 'cumulative_return_percentage': 0.0, 'avg_return_per_trade_percentage': 0.0, 'win_rate_percentage': 0.0, 'profit_loss_ratio': np.nan, 'max_single_trade_drawdown_percentage': 0.0}


            print(
                f"  -> 最佳策略: G{best_metric['generator_index']} (模式 {best_metric['mode']}) | 交易: {best_metric['num_trades']} | 收益: {best_metric['cumulative_return_percentage']:.2f}% | 均笔收益: {best_metric.get('avg_return_per_trade_percentage', 0):.2f}% | 胜率: {best_metric['win_rate_percentage']:.2f}% | 盈亏比: {best_metric.get('profit_loss_ratio', 0):.2f} | 最大单笔回撤: {best_metric.get('max_single_trade_drawdown_percentage', 0):.2f}%"
            )

            if best_metric.get('generator_index', 'N/A') != 'N/A':
                output_dir_stock = os.path.join(FILTERED_OUTPUT_BASE_DIR, sector, stock_name)
                src_base = f"G{best_metric['generator_index']}_mode{best_metric['mode']}"

                for suffix in ["_daily_signals.csv", "_metrics.csv", "_individual_trades_returns.png",
                               "_cumulative_return_drawdown.png"]:
                    src_filename = f"{src_base}{suffix}"
                    dest_filename = src_filename.replace(src_base, "best")

                    src_path = os.path.join(output_dir_stock, src_filename)
                    dest_path = os.path.join(output_dir_stock, dest_filename)

                    if os.path.exists(src_path):
                        try:
                            shutil.copy2(src_path, dest_path)
                            print(f"  已复制: {src_filename} -> {dest_filename}")
                        except Exception as e:
                            print(f"  复制文件出错: {e}")
                    else:
                        if '.csv' in src_path:
                            print(f"  警告: 源文件未找到，无法复制: {src_path}")

    print("=" * 50 + "\n所有任务完成。\n" + "=" * 50)