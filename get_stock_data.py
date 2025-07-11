# get_stock_data.py

import tushare as ts
import pandas as pd
import talib
import os
import time
from datetime import datetime, timedelta
import traceback
from typing import Optional, Tuple, List
import sys
import json
import multiprocessing
import argparse
import numpy as np

# --- 配置区 ---
TUSHARE_TOKEN = '5c9bcf56aeee0f2738748e413c0bd3112e22b0897618eaf7f9b4ca41'  # pro <--- 请务必替换为您的真实Tushare Token!
# 数据存储的根目录
BASE_DATA_DIR = 'csv_data'

# 训练数据和预测数据的子目录
TRAIN_SUBDIR = 'train'
PREDICT_SUBDIR = 'predict'

# --- NEW: 控制是否使用复权数据 ---
USE_ADJUSTED_DATA = True

# --- NEW: 指标计算预热期 ---
# 为确保长周期指标（如240日均线）有足够的数据计算，
# 最终保存的CSV文件将从获取到的数据中跳过此期间的天数。
INDICATOR_WARMUP_PERIOD = 240

# 数据获取时间范围
TRAIN_DATA_FETCH_START_DATE = '20120101'
USER_TRAIN_END_DATE = '20250630'

# 新增:预测数据用户自定义结束日期
PREDICT_DATA_USER_END_DATE = 'latest'

stock_info_dict = {
    #"物联网": ["远望谷", "东信和平"],
    #"军工": ["长城军工", "烽火电子", "中兵红箭"],
    "培育钻石": ["黄河旋风"],
    "港口": ["凤凰航运"],
    #"传媒": ["新华传媒", "吉视传媒"], # 添加了几个股票以便测试多只获取
    # "零售": ["全新好", "永辉超市", "中百集团", "东百集团"],
}

# --- Tushare API 初始化 ---
try:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
except Exception as e:
    print(f"错误: 无法在 get_stock_data.py 中全局初始化 Tushare Pro API 'pro': {e}")
    print("请确保您的 TUSHARE_TOKEN 正确且网络连接正常。")
    pro = None


def get_tushare_pro_instance():
    return ts.pro_api(token=TUSHARE_TOKEN)


# --- 全局交易日历 DataFrame ---
GLOBAL_TRADE_CAL_DF: Optional[pd.DataFrame] = None


# --- 辅助函数 ---
def find_next_trade_day(start_date_str: str, trade_cal_df: pd.DataFrame) -> str:
    current_date = pd.to_datetime(start_date_str, format='%Y%m%d', errors='coerce')
    if pd.isna(current_date):
        raise ValueError(f"Invalid start_date_str provided: {start_date_str}")
    future_cal = trade_cal_df[trade_cal_df['trade_date_dt'] >= current_date].sort_values('trade_date_dt')
    for _, row in future_cal.iterrows():
        if row['is_open'] == 1:
            return row['trade_date']
    raise ValueError(f"Could not find a trade day from {start_date_str} in the pre-loaded trade calendar.")


def find_latest_trading_day_on_or_before(date_dt: datetime, trade_cal_df: pd.DataFrame) -> Optional[datetime]:
    filtered_cal = trade_cal_df[
        (trade_cal_df['trade_date_dt'] <= date_dt) &
        (trade_cal_df['is_open'] == 1)
        ].sort_values('trade_date_dt', ascending=False)
    if not filtered_cal.empty:
        return filtered_cal['trade_date_dt'].iloc[0]
    return None


def create_data_dirs_if_not_exists(sector_name: str, stock_name: str) -> Tuple[str, str]:
    train_stock_dir = os.path.join(BASE_DATA_DIR, TRAIN_SUBDIR, sector_name, stock_name)
    predict_stock_dir = os.path.join(BASE_DATA_DIR, PREDICT_SUBDIR, sector_name, stock_name)
    os.makedirs(train_stock_dir, exist_ok=True)
    os.makedirs(predict_stock_dir, exist_ok=True)
    return train_stock_dir, predict_stock_dir


def get_ts_code_from_name(stock_name: str, stock_basic_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    result = stock_basic_df[stock_basic_df['name'] == stock_name]
    if not result.empty:
        return result.iloc[0]['ts_code'], result.iloc[0]['industry']
    else:
        return None, None


def calculate_technical_indicators(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    # 添加 pctChg 到必需列，因为技术指标可能会用到价格变化率
    required_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'pctChg']
    for col in required_ohlcv_cols:
        if col not in df.columns:
            # 如果缺少 pctChg，可以尝试根据 close 和 pre_close 计算
            if col == 'pctChg' and 'close' in df.columns and 'pre_close' in df.columns:
                df['pctChg'] = (df['close'] / df['pre_close'].shift(1) - 1) * 100  # 假设 pre_close 是前一天的收盘价
                df['pctChg'].iloc[0] = 0  # 第一个交易日的涨跌幅设为0或NaN
            else:
                print(f"警告: 计算技术指标前, DataFrame缺少必需的列: {col}。跳过指标计算。")
                return pd.DataFrame()
        # MODIFIED: 在函数入口再次强制转换，作为双重保险
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 重新定义需要进行dropna的子集，不包括pctChg，因为它在手动计算后可能开头有NaN
    dropna_subset_cols = ['open', 'high', 'low', 'close', 'volume']
    df.dropna(subset=dropna_subset_cols, inplace=True)

    if df.empty: return df

    # 确保所有计算列都是浮点数类型
    op = df['open'].astype(float)
    hi = df['high'].astype(float)
    lo = df['low'].astype(float)
    cl = df['close'].astype(float)
    vo = df['volume'].astype(float)
    pct_change = df['pctChg'].astype(float)  # 使用经过复权或原始的pctChg

    indicators_dict = {}

    try:
        # --- 1. 基础指标 ---
        indicators_dict['sma_5'] = talib.SMA(cl, timeperiod=5)
        indicators_dict['sma_10'] = talib.SMA(cl, timeperiod=10)
        indicators_dict['sma_20'] = talib.SMA(cl, timeperiod=20)
        indicators_dict['sma_30'] = talib.SMA(cl, timeperiod=30)
        indicators_dict['sma_60'] = talib.SMA(cl, timeperiod=60)
        #indicators_dict['sma_120'] = talib.SMA(cl, timeperiod=120)
        #indicators_dict['sma_240'] = talib.SMA(cl, timeperiod=240)
        indicators_dict['ema_5'] = talib.EMA(cl, timeperiod=5)
        indicators_dict['ema_10'] = talib.EMA(cl, timeperiod=10)
        indicators_dict['ema_20'] = talib.EMA(cl, timeperiod=20)
        indicators_dict['ema_30'] = talib.EMA(cl, timeperiod=30)
        indicators_dict['ema_60'] = talib.EMA(cl, timeperiod=60)
        #indicators_dict['ema_120'] = talib.EMA(cl, timeperiod=120)
        #indicators_dict['ema_240'] = talib.EMA(cl, timeperiod=240)

        # 原始MACD (12, 26, 9)
        macd_dif, macd_dea, macd_hist = talib.MACD(cl, fastperiod=12, slowperiod=26, signalperiod=9)
        indicators_dict['macd_12_26_9'] = macd_dif
        indicators_dict['macds_12_26_9'] = macd_dea
        indicators_dict['macdh_12_26_9'] = macd_hist

        # 新增MACD (10, 20, 9)
        macd_dif_10_20, macd_dea_10_20, macd_hist_10_20 = talib.MACD(cl, fastperiod=10, slowperiod=20, signalperiod=9)
        indicators_dict['macd_10_20_9'] = macd_dif_10_20
        indicators_dict['macds_10_20_9'] = macd_dea_10_20
        indicators_dict['macdh_10_20_9'] = macd_hist_10_20

        # KDJ (14, 3, 3) - 经典参数
        stochk_14, stochd_14 = talib.STOCH(hi, lo, cl, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3,
                                           slowd_matype=0)
        indicators_dict['stochk_14_3_3'] = stochk_14
        indicators_dict['stochd_14_3_3'] = stochd_14
        indicators_dict['j_kdj_14_3_3'] = 3 * indicators_dict['stochk_14_3_3'] - 2 * indicators_dict['stochd_14_3_3']

        # KDJ (9, 3, 3) - 更灵敏的参数
        stochk_9, stochd_9 = talib.STOCH(hi, lo, cl, fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3,
                                         slowd_matype=0)
        indicators_dict['stochk_9_3_3'] = stochk_9
        indicators_dict['stochd_9_3_3'] = stochd_9
        indicators_dict['j_kdj_9_3_3'] = 3 * indicators_dict['stochk_9_3_3'] - 2 * indicators_dict['stochd_9_3_3']

        indicators_dict['rsi_14'] = talib.RSI(cl, timeperiod=14)
        indicators_dict['adx_14'] = talib.ADX(hi, lo, cl, timeperiod=14)
        indicators_dict['mfi_14'] = talib.MFI(hi, lo, cl, vo, timeperiod=14)
        indicators_dict['cci_14'] = talib.CCI(hi, lo, cl, timeperiod=14)
        indicators_dict['willr_14'] = talib.WILLR(hi, lo, cl, timeperiod=14)
        indicators_dict['mom_10'] = talib.MOM(cl, timeperiod=10)
        indicators_dict['roc_10'] = talib.ROC(cl, timeperiod=10)

        # --- 2. 波动性与区间指标 ---
        indicators_dict['atr_14'] = talib.ATR(hi, lo, cl, timeperiod=14)
        indicators_dict['trange'] = talib.TRANGE(hi, lo, cl)
        bbu, bbm, bbl = talib.BBANDS(cl, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        indicators_dict['bbu_20_2'] = bbu
        indicators_dict['bbm_20_2'] = bbm
        indicators_dict['bbl_20_2'] = bbl
        indicators_dict['bbw_20_2'] = (indicators_dict['bbu_20_2'] - indicators_dict['bbl_20_2']) / (
                    indicators_dict['bbm_20_2'] + 1e-9)
        indicators_dict['stddev_5d'] = talib.STDDEV(cl, timeperiod=5, nbdev=1)

        # --- 3. 量能指标 ---
        indicators_dict['obv'] = talib.OBV(cl, vo)
        indicators_dict['ad_line'] = talib.AD(hi, lo, cl, vo)



        indicators_df = pd.DataFrame(indicators_dict, index=df.index)
        df_with_indicators = pd.concat([df, indicators_df], axis=1)

    except Exception as e:
        print(f"警告: 计算技术指标时发生错误: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    return df_with_indicators


def add_future_direction_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    (已禁用) 不再生成 'direction' 列。
    此函数现在只返回原始的DataFrame，以保持调用链的完整性，但不起任何作用。
    """
    return df


def robust_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    对DataFrame进行彻底的清理，确保所有数值列都是正确的数字类型。
    这是解决增量更新时数据类型不一致问题的关键。
    """
    if df.empty:
        return df

    df_cleaned = df.copy()

    # 1. 确保日期列是标准格式
    if 'date' in df_cleaned.columns:
        df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y%m%d')
        # 保留可以解析的日期行，无法解析的日期行可能会影响排序和连续性，但这里不再删除
        # 仅删除date为NaT的行
        df_cleaned.dropna(subset=['date'], inplace=True)

    # 2. 定义所有应该为数值类型的列
    numeric_cols = [
        'open', 'high', 'low', 'close', 'volume', 'pre_close', 'pctChg', 'turn'
    ]

    # 3. 强制转换这些列为数值类型，无法转换的设为NaN
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    # 4. 按日期排序并重置索引，确保数据连续性
    df_cleaned = df_cleaned.sort_values(by='date').reset_index(drop=True)

    return df_cleaned


def preprocess_raw_df(df_raw_daily: pd.DataFrame, df_raw_basic: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df_raw_daily.empty: return pd.DataFrame()
    df_daily = df_raw_daily.rename(columns={'vol': 'volume', 'trade_date': 'date', 'pct_chg': 'pctChg'})

    # 确保包含 pre_close，因为它用于后续 pctChg 的重新计算
    daily_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if 'pctChg' in df_daily.columns: daily_cols.append('pctChg')
    if 'pre_close' in df_daily.columns: daily_cols.append('pre_close')

    # 检查是否有重复列名
    if len(set(daily_cols)) != len(daily_cols):
        print(f"警告: preprocess_raw_df 发现重复列名，请检查输入。")
        return pd.DataFrame()

    # 确保只选择存在的列，避免 KeyError
    daily_cols_existing = [col for col in daily_cols if col in df_daily.columns]
    df_daily = df_daily[daily_cols_existing].copy()

    df_merged = df_daily.copy()  # 从 df_daily 复制，而不是重新创建空的 df_merged

    # 合并基本数据 (如果存在且有效)
    if df_raw_basic is not None and not df_raw_basic.empty:
        df_basic_renamed = df_raw_basic.rename(columns={'trade_date': 'date', 'turnover_rate': 'turn'})
        basic_cols = ['date', 'turn']
        basic_cols_existing = [col for col in basic_cols if col in df_basic_renamed.columns]

        if basic_cols_existing:
            df_basic_selected = df_basic_renamed[basic_cols_existing].copy()
            df_basic_selected['date'] = df_basic_selected['date'].astype(str)
            # 确保合并时不会引入重复的date列
            df_merged = pd.merge(df_daily, df_basic_selected, on='date', how='left', suffixes=('', '_drop')).filter(
                regex='^(?!.*_drop)')
        else:
            print(f"警告: 原始基本数据缺少必需的列: {basic_cols}。将不合并这些基本指标。")

    # 强制转换为字符串以确保后续处理一致
    if 'date' in df_merged.columns:
        df_merged['date'] = df_merged['date'].astype(str)

    return df_merged


def fetch_and_process_stock_data(ts_code: str,
                                 stock_industry_name: str,
                                 stock_name: str,
                                 fetch_start_date: str,
                                 predict_end_date_str: str,
                                 user_train_end_date_str: str,
                                 global_trade_cal_df: pd.DataFrame,
                                 pro_instance,
                                 stock_adj_factor_df: Optional[pd.DataFrame] = None
                                 ) -> bool:
    try:
        train_stock_dir, predict_stock_dir = create_data_dirs_if_not_exists(stock_industry_name, stock_name)
        train_filename = f"{ts_code.replace('.', '_')}_{stock_industry_name}_{stock_name}_train_data.csv"
        train_filepath = os.path.join(train_stock_dir, train_filename)
        predict_filename = f"{ts_code.replace('.', '_')}_{stock_industry_name}_{stock_name}_predict_data.csv"
        predict_filepath = os.path.join(predict_stock_dir, predict_filename)
        predict_meta_filepath = predict_filepath.replace('.csv', '.meta')

        current_fetch_start_date_for_new_segment = TRAIN_DATA_FETCH_START_DATE
        df_existing_raw = pd.DataFrame()

        target_end_dt = pd.to_datetime(predict_end_date_str, format='%Y%m%d', errors='coerce')

        is_up_to_date_via_meta = False
        if pd.notna(target_end_dt):
            if os.path.exists(predict_meta_filepath):
                try:
                    with open(predict_meta_filepath, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    last_saved_date_str_from_meta = meta_data.get('last_date')

                    if last_saved_date_str_from_meta:
                        last_saved_date_dt_from_meta = pd.to_datetime(last_saved_date_str_from_meta, format='%Y%m%d',
                                                                      errors='coerce')

                        if pd.notna(last_saved_date_dt_from_meta):
                            if last_saved_date_dt_from_meta >= target_end_dt:
                                print(
                                    f"股票 {stock_name} ({ts_code}): (通过元数据文件) 已存在且数据已达到或超过目标结束日期 {predict_end_date_str}。无需获取新数据。")
                                is_up_to_date_via_meta = True
                            else:
                                latest_trading_day_on_or_before_target = find_latest_trading_day_on_or_before(
                                    target_end_dt, global_trade_cal_df)
                                if latest_trading_day_on_or_before_target is not None and last_saved_date_dt_from_meta >= latest_trading_day_on_or_before_target:
                                    print(
                                        f"股票 {stock_name} ({ts_code}): (通过元数据文件) 已存在且数据已更新至最新交易日 ({last_saved_date_dt_from_meta.strftime('%Y%m%d')})。目标日期 {predict_end_date_str} 为非交易日或节假日,无需获取新数据。")
                                    is_up_to_date_via_meta = True
                                else:
                                    print(
                                        f"股票 {stock_name} ({ts_code}): 元数据文件 '{predict_meta_filepath}' 存在但日期过期或不完整,将回退到完整文件检查。")
                        else:
                            print(
                                f"股票 {stock_name} ({ts_code}): 元数据文件 '{predict_meta_filepath}' 中的日期格式无效,将回退到完整文件检查。")
                    else:
                        print(
                            f"股票 {stock_name} ({ts_code}): 元数据文件 '{predict_meta_filepath}' 缺少 'last_date' 字段,将回退到完整文件检查。")
                except Exception as e:
                    print(
                        f"股票 {stock_name} ({ts_code}): 读取元数据文件 '{predict_meta_filepath}' 失败: {e}。将回退到完整文件检查。")
                    traceback.print_exc()

        if is_up_to_date_via_meta: return True

        # 读取现有数据时，只读取原始数据，不包含指标
        if os.path.exists(predict_filepath):
            try:
                raw_cols_to_read = ['date', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'pctChg', 'turn']

                # 仅选择文件中实际存在的列来读取
                cols_in_file = pd.read_csv(predict_filepath, nrows=0).columns.tolist()
                actual_raw_cols_to_read = [col for col in raw_cols_to_read if col in cols_in_file]

                temp_df = pd.read_csv(predict_filepath, dtype={'date': str}, usecols=actual_raw_cols_to_read)

                if not temp_df.empty and 'date' in temp_df.columns:
                    df_existing_raw = temp_df.copy()
                    df_existing_raw['date_dt_temp'] = pd.to_datetime(df_existing_raw['date'])
                    last_saved_date_dt = df_existing_raw['date_dt_temp'].max()

                    latest_trading_day_on_or_before_target = find_latest_trading_day_on_or_before(target_end_dt,
                                                                                                  global_trade_cal_df)
                    if latest_trading_day_on_or_before_target is not None and last_saved_date_dt >= latest_trading_day_on_or_before_target:
                        print(
                            f"股票 {stock_name} ({ts_code}): '{predict_filename}' 已存在且数据已更新至最新交易日 ({last_saved_date_dt.strftime('%Y%m%d')})。目标日期 {predict_end_date_str} 为非交易日或节假日,无需获取新数据。")
                        current_fetch_start_date_for_new_segment = (target_end_dt + timedelta(days=1)).strftime(
                            '%Y%m%d')
                    else:
                        current_fetch_start_date_for_new_segment = (last_saved_date_dt + timedelta(days=1)).strftime(
                            '%Y%m%d')
                        print(
                            f"股票 {stock_name} ({ts_code}): '{predict_filename}' 已存在,上次更新至 {last_saved_date_dt.strftime('%Y-%m-%d')}。本次从 {current_fetch_start_date_for_new_segment} 获取新数据段至 {predict_end_date_str}。")
                    df_existing_raw.drop(columns=['date_dt_temp'], inplace=True, errors='ignore')
                else:
                    print(
                        f"股票 {stock_name} ({ts_code}): 警告: 现有预测文件 '{predict_filepath}' 为空或缺少原始列,将尝试重新获取全部数据至 {predict_end_date_str}。")
                    df_existing_raw = pd.DataFrame()
            except Exception as e:
                print(
                    f"股票 {stock_name} ({ts_code}): 读取现有预测文件 '{predict_filepath}' 失败: {e},将尝试重新获取全部数据至 {predict_end_date_str}。")
                traceback.print_exc()
                df_existing_raw = pd.DataFrame()

        # --- 确定基准复权因子 ---
        base_adjustment_factor = 1.0
        use_adj_for_this_stock = USE_ADJUSTED_DATA
        if use_adj_for_this_stock and stock_adj_factor_df is not None and not stock_adj_factor_df.empty:
            temp_df = stock_adj_factor_df[stock_adj_factor_df['trade_date'] <= user_train_end_date_str].copy()
            if not temp_df.empty:
                # 按日期降序排列，取第一个，即离基准日最近的那个
                base_adjustment_factor = temp_df.sort_values(by='trade_date', ascending=False).iloc[0]['adj_factor']
                print(f"  -> 复权模式: 已确定基准日 {user_train_end_date_str} 的复权因子为: {base_adjustment_factor}")
            else:
                print(f"  -> 警告: 找不到基准日 {user_train_end_date_str} 的复权因子。将使用不复权数据。")
                use_adj_for_this_stock = False
        elif use_adj_for_this_stock:
            print(f"  -> 警告: 未提供复权因子数据。将使用不复权数据。")
            use_adj_for_this_stock = False

        df_new_processed_segment = pd.DataFrame()
        if pd.to_datetime(current_fetch_start_date_for_new_segment) <= pd.to_datetime(predict_end_date_str):
            try:
                actual_fetch_start_date_for_segment = find_next_trade_day(current_fetch_start_date_for_new_segment,
                                                                          global_trade_cal_df)
                if pd.to_datetime(actual_fetch_start_date_for_segment) <= pd.to_datetime(predict_end_date_str):
                    print(
                        f"股票 {stock_name} ({ts_code}): 正在获取从 {actual_fetch_start_date_for_segment} 到 {predict_end_date_str} 的不复权原始数据...")

                    df_raw_daily_segment_data = pro_instance.daily(
                        ts_code=ts_code,
                        start_date=actual_fetch_start_date_for_segment,
                        end_date=predict_end_date_str
                    )
                    time.sleep(0.2)

                    df_raw_basic_segment_data = pro_instance.daily_basic(
                        ts_code=ts_code,
                        start_date=actual_fetch_start_date_for_segment,
                        end_date=predict_end_date_str
                    )
                    time.sleep(0.2)

                    df_new_processed_segment = preprocess_raw_df(df_raw_daily_segment_data, df_raw_basic_segment_data)
            except Exception as e:
                print(f"股票 {stock_name} ({ts_code}): 获取或处理新数据段时发生错误: {e}")
                traceback.print_exc()
                df_new_processed_segment = pd.DataFrame()

        df_combined_raw = pd.DataFrame()
        if not df_new_processed_segment.empty:
            if not df_existing_raw.empty:
                # 确保合并前按日期排序，再去除重复项
                df_combined_raw = pd.concat([df_existing_raw, df_new_processed_segment]).sort_values(
                    'date').drop_duplicates(subset=['date'], keep='last')
            else:
                df_combined_raw = df_new_processed_segment
        else:
            if not df_existing_raw.empty:
                print(f"股票 {stock_name} ({ts_code}): 未获取到新数据段，将使用现有数据进行处理。")
                df_combined_raw = df_existing_raw
            else:
                print(f"股票 {stock_name} ({ts_code}): 既无现有数据也无新数据。无法处理。")
                return False

        if df_combined_raw.empty:
            print(f"股票 {stock_name} ({ts_code}): 合并后的数据集为空。跳过。")
            return False

        # --- 手动复权计算 ---
        if use_adj_for_this_stock and stock_adj_factor_df is not None and not stock_adj_factor_df.empty:
            print(f"  -> 正在对 {len(df_combined_raw)} 行数据进行手动前复权计算...")

            # 确保合并前日期格式一致
            df_combined_raw['trade_date_dt'] = pd.to_datetime(df_combined_raw['date'], format='%Y%m%d', errors='coerce')
            stock_adj_factor_df['trade_date_dt'] = pd.to_datetime(stock_adj_factor_df['trade_date'], format='%Y%m%d',
                                                                  errors='coerce')

            # 使用左合并，保留所有日线数据日期
            df_with_adj = pd.merge(df_combined_raw, stock_adj_factor_df[['trade_date_dt', 'adj_factor']],
                                   on='trade_date_dt', how='left')
            df_with_adj.sort_values(by='trade_date_dt', inplace=True)

            # 核心修复：填充缺失的复权因子
            # 推荐的修改
            # 1. 先用 bfill() 向后填充，用第一个有效因子填充最开始的 NaN
            df_with_adj['adj_factor'].fillna(method='bfill', inplace=True)

            # 2. 再用 ffill() 向前填充，处理中间的停牌日等 NaN
            df_with_adj['adj_factor'].fillna(method='ffill', inplace=True)

            # 转换为数值类型，errors='coerce' 将非数字转换为 NaN
            for col in ['open', 'high', 'low', 'close', 'pre_close']:
                df_with_adj[col] = pd.to_numeric(df_with_adj[col], errors='coerce')

            df_with_adj['adj_factor'] = pd.to_numeric(df_with_adj['adj_factor'], errors='coerce')

            # 应用公式
            # 注意：这里需要处理 adj_factor 自身也可能是 NaN 的情况 (尽管上面填充了 1.0)
            # 并且要避免除以0或NaN
            valid_adj_mask = (df_with_adj['adj_factor'].notna()) & (base_adjustment_factor != 0) & (
                pd.notna(base_adjustment_factor))

            for col in ['open', 'high', 'low', 'close', 'pre_close']:
                df_with_adj.loc[valid_adj_mask, col] = df_with_adj.loc[valid_adj_mask, col] * df_with_adj.loc[
                    valid_adj_mask, 'adj_factor'] / base_adjustment_factor

            # 重新计算 pct_chg，因为原始 pctChg 是基于不复权价格
            df_with_adj['pre_close_shifted'] = df_with_adj['close'].shift(1)
            valid_pct_chg_mask = (df_with_adj['pre_close_shifted'].notna()) & (
                        df_with_adj['pre_close_shifted'] != 0) & (df_with_adj['close'].notna())
            df_with_adj['pctChg'] = np.nan  # Initialize with NaN
            df_with_adj.loc[valid_pct_chg_mask, 'pctChg'] = (df_with_adj.loc[valid_pct_chg_mask, 'close'] /
                                                             df_with_adj.loc[
                                                                 valid_pct_chg_mask, 'pre_close_shifted'] - 1) * 100
            df_with_adj.drop(columns=['pre_close_shifted'], inplace=True, errors='ignore')

            # 删除临时列
            df_combined_raw = df_with_adj.drop(columns=['trade_date_dt', 'adj_factor'])

        print(f"股票 {stock_name} ({ts_code}): 正在对合并后的 {len(df_combined_raw)} 行完整数据进行彻底清洗...")
        df_cleaned_full = robust_clean_dataframe(df_combined_raw)

        if df_cleaned_full.empty:
            print(f"股票 {stock_name} ({ts_code}): 清洗后数据集为空。跳过。")
            return False

        print(f"股票 {stock_name} ({ts_code}): 在清洗后的完整数据上重新计算所有技术指标...")
        # calculate_technical_indicators 现在会使用复权或不复权后的OHLCV和重新计算的pctChg
        df_with_indicators_full = calculate_technical_indicators(df_cleaned_full)

        if df_with_indicators_full.empty:
            print(f"股票 {stock_name} ({ts_code}): 在计算指标后数据集为空。无法保存文件。")
            return False

        df_processed_final = df_with_indicators_full

        if len(df_processed_final) > INDICATOR_WARMUP_PERIOD:
            # Ensure we slice after sorting
            df_processed_final = df_processed_final.sort_values('date').iloc[INDICATOR_WARMUP_PERIOD:].reset_index(
                drop=True)
            print(
                f"股票 {stock_name} ({ts_code}): 已应用 {INDICATOR_WARMUP_PERIOD} 天的指标预热期，剩余 {len(df_processed_final)} 条有效数据。")
        else:
            print(
                f"股票 {stock_name} ({ts_code}): 警告: 数据总行数 ({len(df_processed_final)}) 不足以应用预热期 ({INDICATOR_WARMUP_PERIOD})。")

        if df_processed_final.empty:
            print(f"股票 {stock_name} ({ts_code}): 应用预热期后数据集为空。无法保存文件。")
            return False

        df_to_save_predict = df_processed_final.copy()
        df_to_save_predict = df_to_save_predict[
            pd.to_datetime(df_to_save_predict['date']) <= pd.to_datetime(predict_end_date_str)]
        df_to_save_predict['date'] = pd.to_datetime(df_to_save_predict['date']).dt.strftime('%Y%m%d')
        print(
            f"股票 {stock_name} ({ts_code}): 预测数据 (至 {predict_end_date_str}) 已保存到: {predict_filepath} (共 {len(df_to_save_predict)} 条)")
        df_to_save_predict.to_csv(predict_filepath, index=False, na_rep='')

        if not df_to_save_predict.empty:
            latest_date_in_saved_df = df_to_save_predict['date'].max()
            meta_data_to_save = {"last_date": latest_date_in_saved_df}
            try:
                with open(predict_meta_filepath, 'w', encoding='utf-8') as f:
                    json.dump(meta_data_to_save, f)
                print(f"股票 {stock_name} ({ts_code}): 元数据文件已保存到: {predict_meta_filepath}")
            except Exception as e:
                print(f"股票 {stock_name} ({ts_code}): 保存元数据文件 '{predict_meta_filepath}' 失败: {e}")
        else:
            print(f"股票 {stock_name} ({ts_code}): 警告: '{predict_filepath}' 为空,未创建元数据文件。")

        df_train_to_save = df_processed_final[
            pd.to_datetime(df_processed_final['date']) <= pd.to_datetime(user_train_end_date_str)].copy()
        df_train_to_save['date'] = pd.to_datetime(df_train_to_save['date']).dt.strftime('%Y%m%d')
        if not df_train_to_save.empty:
            df_train_to_save.to_csv(train_filepath, index=False, na_rep='')
            print(
                f"股票 {stock_name} ({ts_code}): 训练数据 (至 {user_train_end_date_str}) 已保存到: {train_filepath} (共 {len(df_train_to_save)} 条)")
        else:
            print(
                f"股票 {stock_name} ({ts_code}): 在训练结束日期 {user_train_end_date_str} 之前没有可用数据。训练文件未创建。")

        return True
    except Exception as e:
        print(f"股票 {stock_name} ({ts_code}): 处理过程中发生未知错误: {e}")
        traceback.print_exc()
        return False


def _worker_fetch_data(stock_item: dict, global_trade_cal_df: pd.DataFrame, all_stock_basic_df: pd.DataFrame,
                       predict_end_date_str: str, all_adj_factors_df: Optional[pd.DataFrame] = None) -> dict:
    _pro_instance = get_tushare_pro_instance()
    stock_name = stock_item['name']
    sector = stock_item['sector']
    ts_code, stock_industry_name = get_ts_code_from_name(stock_name, all_stock_basic_df)
    if not ts_code:
        return {'stock_name': stock_name, 'ts_code': 'N/A', 'sector': sector, 'status': 'skipped_no_ts_code'}

    stock_adj_factor_df = None
    if all_adj_factors_df is not None and not all_adj_factors_df.empty:
        stock_adj_factor_df = all_adj_factors_df[all_adj_factors_df['ts_code'] == ts_code].copy()

    filename_sector = stock_industry_name if stock_industry_name else "未知板块"
    success = fetch_and_process_stock_data(ts_code, filename_sector, stock_name, TRAIN_DATA_FETCH_START_DATE,
                                           predict_end_date_str, USER_TRAIN_END_DATE, global_trade_cal_df,
                                           _pro_instance, stock_adj_factor_df)
    return {'stock_name': stock_name, 'ts_code': ts_code, 'sector': sector,
            'status': 'success' if success else 'failed'}


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="并行获取和处理股票数据。")
    parser.add_argument('--num_cores', type=int, default=min(32, multiprocessing.cpu_count()),
                        help=f"指定用于并行获取的CPU核心数。默认为32核或系统核心数(取小值)({min(32, multiprocessing.cpu_count())})。")
    args = parser.parse_args()

    DATA_FETCH_LATEST_END_DATE = ''
    if PREDICT_DATA_USER_END_DATE is None or str(PREDICT_DATA_USER_END_DATE).lower() == 'latest':
        DATA_FETCH_LATEST_END_DATE = datetime.now().strftime('%Y%m%d')
        print(f"预测数据将获取到最新日期: {DATA_FETCH_LATEST_END_DATE}")
    else:
        user_provided_date_str = str(PREDICT_DATA_USER_END_DATE)
        try:
            datetime.strptime(user_provided_date_str, '%Y%m%d')
            DATA_FETCH_LATEST_END_DATE = user_provided_date_str
            print(f"预测数据将获取到固定日期: {DATA_FETCH_LATEST_END_DATE}")
        except ValueError:
            print(f"警告: PREDICT_DATA_USER_END_DATE ('{user_provided_date_str}') 格式无效。将默认使用最新日期。")
            DATA_FETCH_LATEST_END_DATE = datetime.now().strftime('%Y%m%d')

    print("正在获取所有A股基本信息以匹配股票名称...")
    try:
        all_stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry')
        if all_stock_basic.empty:
            print("错误:未能获取股票基本信息列表,请检查Tushare连接或Token。")
            sys.exit(1)
    except Exception as e:
        print(f"错误:获取股票基本信息失败: {e}")
        traceback.print_exc()
        sys.exit(1)
    print("股票基本信息获取完毕。")

    print("正在加载交易日历...")
    calendar_start_date = TRAIN_DATA_FETCH_START_DATE
    calendar_end_date = (datetime.strptime(DATA_FETCH_LATEST_END_DATE, '%Y%m%d') + timedelta(days=365)).strftime(
        '%Y%m%d')
    try:
        GLOBAL_TRADE_CAL_DF = pro.trade_cal(exchange='SSE', start_date=calendar_start_date, end_date=calendar_end_date)
        if GLOBAL_TRADE_CAL_DF.empty:
            print("错误:未能获取交易日历,请检查Tushare连接或Token。")
            sys.exit(1)
        if 'cal_date' in GLOBAL_TRADE_CAL_DF.columns:
            GLOBAL_TRADE_CAL_DF.rename(columns={'cal_date': 'trade_date'}, inplace=True)
        else:
            print("警告: Tushare交易日历返回的DataFrame中未找到'cal_date'列。请检查Tushare API文档或数据结构。")
            sys.exit(1)
        GLOBAL_TRADE_CAL_DF['trade_date_dt'] = pd.to_datetime(GLOBAL_TRADE_CAL_DF['trade_date'], format='%Y%m%d')
        GLOBAL_TRADE_CAL_DF['trade_date_str'] = GLOBAL_TRADE_CAL_DF['trade_date'].astype(str)
        GLOBAL_TRADE_CAL_DF['is_open'] = pd.to_numeric(GLOBAL_TRADE_CAL_DF['is_open'], errors='coerce')
    except Exception as e:
        print(f"错误:获取交易日历失败: {e}")
        traceback.print_exc()
        sys.exit(1)
    print(f"交易日历加载完毕 (从 {calendar_start_date} 到 {calendar_end_date})。")

    stocks_to_process = []
    for sector_from_dict, stock_names_list in stock_info_dict.items():
        for stock_name in stock_names_list: stocks_to_process.append({'name': stock_name, 'sector': sector_from_dict})
    if not stocks_to_process:
        print("没有找到任何需要处理的股票。程序退出。")
        sys.exit(0)

    # --- 一次性获取所有待处理股票的复权因子 ---
    all_adj_factors_df = None
    # 检查 USE_ADJUSTED_DATA 是否为 True 且 pro 对象已成功初始化
    if USE_ADJUSTED_DATA and pro is not None:
        print("\n正在获取所有待处理股票的复权因子（这可能需要一些时间）...")
        ts_codes_to_fetch = []
        for stock_item in stocks_to_process:
            # Corrected unpacking here
            ts_code, _ = get_ts_code_from_name(stock_item['name'], all_stock_basic)
            if ts_code:
                ts_codes_to_fetch.append(ts_code)

        if ts_codes_to_fetch:
            try:
                all_adj_factors_df = pro.adj_factor(
                    ts_code=','.join(ts_codes_to_fetch),
                    start_date=TRAIN_DATA_FETCH_START_DATE,
                    end_date=DATA_FETCH_LATEST_END_DATE
                )
                print(f"成功获取 {len(all_adj_factors_df)} 条复权因子记录。")
            except Exception as e:
                print(f"警告：获取复权因子失败: {e}。将继续使用不复权数据。")
                USE_ADJUSTED_DATA = False  # 获取失败则降级为不复权
        else:
            print("警告：没有找到需要获取复权因子的股票 TS 代码。将继续使用不复权数据。")
            USE_ADJUSTED_DATA = False  # 没有ts_code则降级

    total_stocks_count = len(stocks_to_process)
    num_cores_to_use = min(args.num_cores, total_stocks_count)
    if num_cores_to_use == 0:
        print("没有足够的任务需要并行处理。")
        sys.exit(0)

    print(f"\n--- 开始并行处理 {total_stocks_count} 支股票数据,使用 {num_cores_to_use} 个进程 ---")
    tasks = []
    for stock_item in stocks_to_process:
        tasks.append(
            (stock_item, GLOBAL_TRADE_CAL_DF, all_stock_basic, DATA_FETCH_LATEST_END_DATE, all_adj_factors_df)
        )

    successful_count = 0
    failed_stocks = []
    skipped_ts_code_stocks = []
    with multiprocessing.Pool(processes=num_cores_to_use) as pool:
        results = list(pool.starmap(_worker_fetch_data, tasks))

    for res in results:
        if res['status'] == 'success':
            successful_count += 1
        elif res['status'] == 'failed':
            failed_stocks.append(f"{res['stock_name']} ({res['ts_code']})")
        elif res['status'] == 'skipped_no_ts_code':
            skipped_ts_code_stocks.append(res['stock_name'])

    print("\n--- 股票数据获取与处理完成总结 ---")
    print(f"总计尝试处理股票数: {total_stocks_count}")
    print(f"成功处理股票数: {successful_count}")
    print(f"失败股票数: {len(failed_stocks)}")
    if failed_stocks: print(f"  失败股票列表: {', '.join(failed_stocks)}")
    print(f"跳过股票数 (未找到ts_code): {len(skipped_ts_code_stocks)}")
    if skipped_ts_code_stocks: print(f"  跳过股票列表 (未找到ts_code): {', '.join(skipped_ts_code_stocks)}")
    print("所有计算详情请查看对应的日志输出。")
    print("脚本执行完毕。")