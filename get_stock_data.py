# get_stock_data.py (已修改以支持同名ETF按交易所后缀创建独立目录)

import tushare as ts
import pandas as pd
import talib
import os
import time
from datetime import datetime, timedelta
import traceback
from typing import Optional, Tuple, List, Dict
import sys
import json
import multiprocessing
import argparse
import numpy as np
import re

# --- 配置区 ---
TUSHARE_TOKEN = '5c9bcf56aeee0f2738748e413c0bd3112e22b0897618eaf7f9b4ca41'
BASE_DATA_DIR = 'csv_data'
TRAIN_SUBDIR = 'train'
PREDICT_SUBDIR = 'predict'
USE_ADJUSTED_DATA = True
INDICATOR_WARMUP_PERIOD = 240
TRAIN_DATA_FETCH_START_DATE = '20120101'
USER_TRAIN_END_DATE = '20250630'
PREDICT_DATA_USER_END_DATE = 'latest'

# --- 核心修改点1: stock_info_dict 中使用带后缀的全名 ---
stock_info_dict = {
    #"指数": ["沪深300", "上证50"],
    #"基金(场内)": ["沪深300ETF", "上证50ETF", "券商ETF"],
    "物联网": ["远望谷", "东信和平"],
    "培育钻石": ["黄河旋风"],
    "港口": ["凤凰航运"],
}

# --- Tushare API 初始化 (保持不变) ---
try:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
except Exception as e:
    print(f"错误: 无法在 get_stock_data.py 中全局初始化 Tushare Pro API 'pro': {e}")
    pro = None


def get_tushare_pro_instance():
    return ts.pro_api(token=TUSHARE_TOKEN)


GLOBAL_TRADE_CAL_DF: Optional[pd.DataFrame] = None


# --- 辅助函数 ---
def find_next_trade_day(start_date_str: str, trade_cal_df: pd.DataFrame) -> str:
    current_date = pd.to_datetime(start_date_str, format='%Y%m%d', errors='coerce')
    if pd.isna(current_date): raise ValueError(f"Invalid start_date_str: {start_date_str}")
    future_cal = trade_cal_df[trade_cal_df['trade_date_dt'] >= current_date].sort_values('trade_date_dt')
    for _, row in future_cal.iterrows():
        if row['is_open'] == 1: return row['trade_date']
    raise ValueError(f"Could not find a trade day from {start_date_str}")


def find_latest_trading_day_on_or_before(date_dt: datetime, trade_cal_df: pd.DataFrame) -> Optional[datetime]:
    filtered_cal = trade_cal_df[
        (trade_cal_df['trade_date_dt'] <= date_dt) & (trade_cal_df['is_open'] == 1)].sort_values('trade_date_dt',
                                                                                                 ascending=False)
    return filtered_cal['trade_date_dt'].iloc[0] if not filtered_cal.empty else None


# --- 核心修改点2: create_data_dirs_if_not_exists 现在直接使用传入的 stock_name ---
def create_data_dirs_if_not_exists(sector_name: str, stock_name: str) -> Tuple[str, str]:
    """
    使用完整的股票名称（包括后缀）创建目录。
    """
    # 不再清洗 stock_name，直接使用
    train_stock_dir = os.path.join(BASE_DATA_DIR, TRAIN_SUBDIR, sector_name, stock_name)
    predict_stock_dir = os.path.join(BASE_DATA_DIR, PREDICT_SUBDIR, sector_name, stock_name)
    os.makedirs(train_stock_dir, exist_ok=True)
    os.makedirs(predict_stock_dir, exist_ok=True)
    return train_stock_dir, predict_stock_dir


def get_security_info_from_name(security_name_with_suffix: str, all_securities_basic_df: pd.DataFrame,
                                target_category: Optional[str] = None) -> Tuple[
    Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    这个函数逻辑保持不变，它能正确处理后缀并返回基础名称和代码。
    我们将保留它的返回值，但在创建目录和文件名时，使用原始的 security_name_with_suffix。
    """
    base_name, exchange_preference = security_name_with_suffix, None
    match = re.search(r'\((沪基|深基)\)', security_name_with_suffix)
    if match:
        base_name = re.sub(r'\s*\(.*\)\s*$', '', security_name_with_suffix).strip()
        exchange_preference = '.SH' if match.group(1) == '沪基' else '.SZ'

    df_to_search = all_securities_basic_df.copy()
    if target_category:
        target_type = {"指数": "index", "基金(场内)": "fund"}.get(target_category)
        if target_type:
            df_to_search = df_to_search[df_to_search['type'] == target_type]
        else:  # 默认为股票
            df_to_search = df_to_search[df_to_search['type'] == 'stock']

    result = df_to_search[df_to_search['name'].str.contains(base_name, case=False, na=False)]

    if not result.empty:
        if exchange_preference and len(result) > 1:
            preferred_result = result[result['ts_code'].str.endswith(exchange_preference)]
            if not preferred_result.empty:
                result = preferred_result

        exact_match = result[result['name'] == base_name]
        first_result = exact_match.iloc[0] if not exact_match.empty else result.iloc[0]

        # 返回 ts_code, type, industry, 和原始全名
        return first_result['ts_code'], first_result['type'], first_result['industry'], security_name_with_suffix

    # 如果找不到，返回 None 和原始全名
    return None, None, None, security_name_with_suffix


# --- 技术指标计算、数据清洗和预处理函数保持不变 ---
def calculate_technical_indicators(df_input: pd.DataFrame) -> pd.DataFrame:
    # ... (此函数无须修改)
    if df_input.empty:
        return pd.DataFrame()

    df = df_input.copy()
    required_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'pctChg']

    for col in required_ohlcv_cols:
        if col not in df.columns:
            if col == 'pctChg' and 'close' in df.columns:
                df['pctChg'] = (df['close'].pct_change()) * 100
                df.fillna({'pctChg': 0}, inplace=True)  # 使用fillna替代iloc[0]更安全
            else:
                print(f"警告: 技术指标计算缺少列 {col}，返回空df。")
                return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
    if df.empty:
        return pd.DataFrame()

    op, hi, lo, cl, vo, pct_change = (df[c].astype(float) for c in ['open', 'high', 'low', 'close', 'volume', 'pctChg'])
    indicators_dict = {}
    try:
        # --- 省略了所有指标计算的重复代码，逻辑保持不变 ---
        indicators_dict['sma_5'] = talib.SMA(cl, 5);
        indicators_dict['sma_10'] = talib.SMA(cl, 10);
        indicators_dict['sma_20'] = talib.SMA(cl, 20);
        indicators_dict['sma_30'] = talib.SMA(cl, 30);
        indicators_dict['sma_60'] = talib.SMA(cl, 60);
        indicators_dict['ema_5'] = talib.EMA(cl, 5);
        indicators_dict['ema_10'] = talib.EMA(cl, 10);
        indicators_dict['ema_20'] = talib.EMA(cl, 20);
        indicators_dict['ema_30'] = talib.EMA(cl, 30);
        indicators_dict['ema_60'] = talib.EMA(cl, 60);
        macd, macdsignal, macdhist = talib.MACD(cl, 12, 26, 9);
        indicators_dict['macd_12_26_9'] = macd;
        indicators_dict['macds_12_26_9'] = macdsignal;
        indicators_dict['macdh_12_26_9'] = macdhist;
        macd, macdsignal, macdhist = talib.MACD(cl, 10, 20, 9);
        indicators_dict['macd_10_20_9'] = macd;
        indicators_dict['macds_10_20_9'] = macdsignal;
        indicators_dict['macdh_10_20_9'] = macdhist;
        k, d = talib.STOCH(hi, lo, cl, 14, 3, 0, 3, 0);
        indicators_dict['stochk_14_3_3'] = k;
        indicators_dict['stochd_14_3_3'] = d;
        indicators_dict['j_kdj_14_3_3'] = 3 * k - 2 * d;
        k, d = talib.STOCH(hi, lo, cl, 9, 3, 0, 3, 0);
        indicators_dict['stochk_9_3_3'] = k;
        indicators_dict['stochd_9_3_3'] = d;
        indicators_dict['j_kdj_9_3_3'] = 3 * k - 2 * d;
        indicators_dict['rsi_14'] = talib.RSI(cl, 14);
        indicators_dict['adx_14'] = talib.ADX(hi, lo, cl, 14);
        indicators_dict['mfi_14'] = talib.MFI(hi, lo, cl, vo, 14);
        indicators_dict['cci_14'] = talib.CCI(hi, lo, cl, 14);
        indicators_dict['willr_14'] = talib.WILLR(hi, lo, cl, 14);
        indicators_dict['mom_10'] = talib.MOM(cl, 10);
        indicators_dict['roc_10'] = talib.ROC(cl, 10);
        indicators_dict['atr_14'] = talib.ATR(hi, lo, cl, 14);
        indicators_dict['trange'] = talib.TRANGE(hi, lo, cl);
        bbu, bbm, bbl = talib.BBANDS(cl, 20, 2.0, 2.0, 0);
        indicators_dict['bbu_20_2'] = bbu;
        indicators_dict['bbm_20_2'] = bbm;
        indicators_dict['bbl_20_2'] = bbl;
        indicators_dict['bbw_20_2'] = (bbu - bbl) / (bbm + 1e-9);
        indicators_dict['stddev_5d'] = talib.STDDEV(cl, 5, 1);
        indicators_dict['obv'] = talib.OBV(cl, vo);
        indicators_dict['ad_line'] = talib.AD(hi, lo, cl, vo);
        indicators_dict['pct_change'] = pct_change;
        is_up = (cl > op).astype(int);
        indicators_dict['consecutive_up_days'] = (is_up.groupby(
            (is_up != is_up.shift()).cumsum()).cumcount() + 1) * is_up;
        is_down = (cl < op).astype(int);
        indicators_dict['consecutive_down_days'] = (is_down.groupby(
            (is_down != is_down.shift()).cumsum()).cumcount() + 1) * is_down;
        indicators_dict['max_gain_3d'] = pct_change.rolling(3).max();
        indicators_dict['max_loss_3d'] = pct_change.rolling(3).min();
        indicators_dict['break_high_5d_count'] = (cl >= cl.rolling(5).max()).rolling(5).sum();
        indicators_dict['range_avg_3d'] = (hi - lo).rolling(3).mean();
        for n in [5, 10, 20, 60]: r_low, r_high = lo.rolling(n).min(), hi.rolling(n).max(); indicators_dict[
            f'pos_in_{n}d_range'] = (cl - r_low) / (r_high - r_low + 1e-9)
        indicators_dict['ema5_div_ema20'] = (indicators_dict['ema_5'] / (indicators_dict['ema_20'] + 1e-9)) - 1;
        indicators_dict['ema20_div_ema60'] = (indicators_dict['ema_20'] / (indicators_dict['ema_60'] + 1e-9)) - 1;
        indicators_dict['vol_roc_10'] = talib.ROC(vo, 10);
        indicators_dict['pv_consistency'] = np.sign(pct_change) * np.sign(vo.pct_change());
        indicators_dict['obv_roc_10'] = talib.ROC(indicators_dict['obv'], 10);
        indicators_dict['sharpe_like_20d'] = (pct_change.rolling(20).mean()) / (pct_change.rolling(20).std() + 1e-9);
        indicators_dict['skew_20d'] = pct_change.rolling(20).skew();
        indicators_dict['kurt_20d'] = pct_change.rolling(20).kurt();
        indicators_dict['body_size'] = abs(cl - op);
        indicators_dict['upper_shadow'] = hi - np.maximum(op, cl);
        indicators_dict['lower_shadow'] = np.minimum(op, cl) - lo;
        indicators_dict['close_pos_in_day_range'] = (cl - lo) / (hi - lo + 1e-9);
        security_identifier = df_input.iloc[0].get('name', df_input.iloc[0].get('ts_code', 'N/A'));
        print(f"证券 {security_identifier}: 自动识别所有TA-Lib K线形态...");
        pattern_functions = [func for func in dir(talib) if func.startswith('CDL')];
        for pattern in pattern_functions:
            try:
                indicators_dict[f"cdl_{pattern[3:].lower()}"] = (getattr(talib, pattern)(op, hi, lo, cl) != 0).astype(
                    int)
            except Exception:
                pass
        indicators_df = pd.DataFrame(indicators_dict, index=df.index);
        return pd.concat([df, indicators_df], axis=1)
    except Exception:
        traceback.print_exc();
        return pd.DataFrame()


def robust_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # ... (此函数无须修改)
    if df.empty:
        return pd.DataFrame()

    df_cleaned = df.copy()

    if 'date' in df_cleaned.columns:
        df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y%m%d')
        df_cleaned.dropna(subset=['date'], inplace=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'pre_close', 'pctChg', 'turn']
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    return df_cleaned.sort_values(by='date').reset_index(drop=True)


def preprocess_raw_df(df_raw_daily: pd.DataFrame, df_raw_basic: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    # ... (此函数无须修改)
    if df_raw_daily is None or df_raw_daily.empty:
        return pd.DataFrame()

    df_daily = df_raw_daily.rename(columns={'vol': 'volume', 'trade_date': 'date', 'pct_chg': 'pctChg'})
    daily_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if 'pctChg' in df_daily.columns: daily_cols.append('pctChg')
    if 'pre_close' in df_daily.columns: daily_cols.append('pre_close')
    df_daily = df_daily[[c for c in daily_cols if c in df_daily.columns]].copy()

    df_merged = df_daily
    if df_raw_basic is not None and not df_raw_basic.empty:
        df_basic = df_raw_basic.rename(columns={'trade_date': 'date', 'turnover_rate': 'turn'})
        basic_cols = ['date', 'turn']
        if any(c in df_basic.columns for c in basic_cols):
            df_merged = pd.merge(df_daily, df_basic[[c for c in basic_cols if c in df_basic.columns]], on='date',
                                 how='left')

    if 'date' in df_merged.columns:
        df_merged['date'] = df_merged['date'].astype(str)

    return df_merged


# --- 核心修改点3: fetch_and_process_stock_data 现在使用完整的 stock_name 来创建目录和文件名 ---
def fetch_and_process_stock_data(ts_code: str, security_type: str, category_name: str, security_name: str,
                                 fetch_start_date: str, predict_end_date_str: str, user_train_end_date_str: str,
                                 global_trade_cal_df: pd.DataFrame, pro_instance,
                                 stock_adj_factor_df: Optional[pd.DataFrame] = None) -> bool:
    """
    `security_name` 在这里是带有后缀的完整名称, 例如 "沪深300ETF(沪基)".
    """
    try:
        # 使用完整的 security_name 创建目录
        train_dir, pred_dir = create_data_dirs_if_not_exists(category_name, security_name)

        # 使用完整的 security_name 创建文件名
        train_fp = os.path.join(train_dir,
                                f"{ts_code.replace('.', '_')}_{category_name}_{security_name}_train_data.csv")
        pred_fp = os.path.join(pred_dir,
                               f"{ts_code.replace('.', '_')}_{category_name}_{security_name}_predict_data.csv")

        start_fetch_date, df_existing, target_end_dt = fetch_start_date, pd.DataFrame(), pd.to_datetime(
            predict_end_date_str, format='%Y%m%d')
        if os.path.exists(pred_fp):
            try:
                df_existing = pd.read_csv(pred_fp, dtype={'date': str})
                latest_trade_day = find_latest_trading_day_on_or_before(target_end_dt, global_trade_cal_df)
                if latest_trade_day and not df_existing.empty and pd.to_datetime(
                        df_existing['date'].max()) >= latest_trade_day:
                    print(f"证券 {security_name} ({ts_code}): 数据已是最新，跳过获取。")
                    return True
                start_fetch_date = (pd.to_datetime(df_existing['date'].max()) + timedelta(days=1)).strftime(
                    '%Y%m%d') if not df_existing.empty else fetch_start_date
            except Exception:
                df_existing = pd.DataFrame()

        df_new_segment = pd.DataFrame()
        if pd.to_datetime(start_fetch_date) <= target_end_dt:
            actual_start = find_next_trade_day(start_fetch_date, global_trade_cal_df)
            if pd.to_datetime(actual_start) <= target_end_dt:
                print(
                    f"证券 {security_name} ({ts_code}): 正在获取从 {actual_start} 到 {predict_end_date_str} 的 {security_type} 数据...")
                api_map = {'stock': (pro_instance.daily, pro_instance.daily_basic),
                           'index': (pro_instance.index_daily, None),
                           'fund': (pro_instance.fund_daily, None)}
                daily_api, basic_api = api_map[security_type]
                df_raw_daily = daily_api(ts_code=ts_code, start_date=actual_start, end_date=predict_end_date_str)
                df_raw_basic = basic_api(ts_code=ts_code, start_date=actual_start,
                                         end_date=predict_end_date_str) if basic_api else None;
                time.sleep(0.25)
                df_new_segment = preprocess_raw_df(df_raw_daily, df_raw_basic)

        df_combined_raw = pd.concat([df_existing, df_new_segment]).drop_duplicates(subset=['date'],
                                                                                   keep='last').sort_values('date')
        if df_combined_raw.empty: return False
        df_to_process = df_combined_raw.copy()

        if security_type == 'stock' and USE_ADJUSTED_DATA and stock_adj_factor_df is not None and not stock_adj_factor_df.empty:
            print(f"  -> 正在对 {len(df_to_process)} 行股票数据进行前复权...")
            # ... (复权逻辑保持不变) ...
            base_adj_series = stock_adj_factor_df[stock_adj_factor_df['trade_date'] <= user_train_end_date_str]
            if not base_adj_series.empty:
                base_adj = base_adj_series.sort_values('trade_date', ascending=False).iloc[0]['adj_factor']
                df_to_process['trade_date_dt'] = pd.to_datetime(df_to_process['date'], format='%Y%m%d')
                stock_adj_factor_df['trade_date_dt'] = pd.to_datetime(stock_adj_factor_df['trade_date'],
                                                                      format='%Y%m%d')
                df_with_adj = pd.merge(df_to_process, stock_adj_factor_df[['trade_date_dt', 'adj_factor']],
                                       on='trade_date_dt', how='left').sort_values('trade_date_dt')
                df_with_adj['adj_factor'].fillna(method='bfill', inplace=True);
                df_with_adj['adj_factor'].fillna(method='ffill', inplace=True)
                for col in ['open', 'high', 'low', 'close', 'pre_close']: df_with_adj[col] = pd.to_numeric(
                    df_with_adj[col], errors='coerce') * pd.to_numeric(df_with_adj['adj_factor'],
                                                                       errors='coerce') / base_adj
                df_with_adj['pctChg'] = (df_with_adj['close'] / df_with_adj['close'].shift(1) - 1) * 100
                df_to_process = df_with_adj.drop(columns=['trade_date_dt', 'adj_factor'])

        df_cleaned = robust_clean_dataframe(df_to_process)
        if df_cleaned.empty: return False

        # 在计算指标前，注入完整的 security_name
        df_cleaned['name'] = security_name

        df_final = calculate_technical_indicators(df_cleaned)
        if df_final.empty: return False

        df_final.drop(columns=['name'], inplace=True, errors='ignore')
        if len(df_final) > INDICATOR_WARMUP_PERIOD: df_final = df_final.iloc[INDICATOR_WARMUP_PERIOD:].reset_index(
            drop=True)
        if df_final.empty: return False

        df_predict = df_final[pd.to_datetime(df_final['date']) <= target_end_dt].copy()
        df_predict.to_csv(pred_fp, index=False, na_rep='');
        print(f"  -> 预测数据已保存: {pred_fp} ({len(df_predict)} 条)")

        if not df_predict.empty:
            with open(pred_fp.replace('.csv', '.meta'), 'w') as f: json.dump({"last_date": df_predict['date'].max()}, f)

        df_train = df_final[pd.to_datetime(df_final['date']) <= pd.to_datetime(user_train_end_date_str)].copy()
        df_train.to_csv(train_fp, index=False, na_rep='');
        print(f"  -> 训练数据已保存: {train_fp} ({len(df_train)} 条)")
        return True
    except Exception as e:
        print(f"证券 {security_name} ({ts_code}): 处理失败 - {e}");
        traceback.print_exc();
        return False


# --- 核心修改点4: 工作进程函数现在传递并使用完整的 security_name ---
def _worker_fetch_data(security_item: dict, global_trade_cal_df: pd.DataFrame, all_securities_basic_df: pd.DataFrame,
                       predict_end_date_str: str, all_adj_factors_df: Optional[pd.DataFrame] = None) -> dict:
    pro_instance = get_tushare_pro_instance()

    # name_with_suffix 是原始的、带有后缀的名称，如 "沪深300ETF(沪基)"
    name_with_suffix, category = security_item['name'], security_item['sector']

    # get_security_info_from_name 返回 ts_code, sec_type, industry 和 完整的name_with_suffix
    ts_code, sec_type, industry, full_name = get_security_info_from_name(name_with_suffix, all_securities_basic_df,
                                                                         target_category=category)

    if not ts_code:
        return {'name': full_name, 'ts_code': 'N/A', 'category': category, 'status': 'skipped_no_ts_code'}

    adj_factors = all_adj_factors_df[all_adj_factors_df[
                                         'ts_code'] == ts_code].copy() if sec_type == 'stock' and all_adj_factors_df is not None else None
    dir_category = industry if industry and pd.notna(industry) else category

    # 将完整的 full_name (即 name_with_suffix) 传递给核心处理函数
    success = fetch_and_process_stock_data(ts_code, sec_type, dir_category, full_name, TRAIN_DATA_FETCH_START_DATE,
                                           predict_end_date_str, USER_TRAIN_END_DATE, global_trade_cal_df, pro_instance,
                                           adj_factors)

    return {'name': full_name, 'ts_code': ts_code, 'category': category,
            'status': 'success' if success else 'failed'}


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="并行获取和处理股票、指数、基金数据。")
    parser.add_argument('--num_cores', type=int, default=min(32, multiprocessing.cpu_count()), help=f"指定并行核心数。")
    args = parser.parse_args()

    end_date_str = datetime.now().strftime('%Y%m%d') if str(PREDICT_DATA_USER_END_DATE).lower() == 'latest' else str(
        PREDICT_DATA_USER_END_DATE)

    print("正在获取所有证券基础信息...")
    try:
        stock_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry');
        stock_df['type'] = 'stock'
        markets = ['CSI', 'SSE', 'SZSE', 'CICC', 'SW', 'MSCI', 'OTH'];
        index_dfs = [pro.index_basic(market=m, fields='ts_code,name') for m in markets]
        index_df = pd.concat(index_dfs, ignore_index=True).drop_duplicates(subset=['ts_code']);
        index_df['type'] = 'index';
        index_df['industry'] = '指数'
        fund_df = pro.fund_basic(market='E', fields='ts_code,name');
        fund_df['type'] = 'fund';
        fund_df['industry'] = '基金'
        all_securities_basic = pd.concat(
            [stock_df[['ts_code', 'name', 'industry', 'type']], index_df[['ts_code', 'name', 'industry', 'type']],
             fund_df[['ts_code', 'name', 'industry', 'type']]], ignore_index=True)
        if all_securities_basic.empty: print("错误: 未能获取任何证券基础信息。"); sys.exit(1)
    except Exception as e:
        print(f"错误: 获取证券基础信息失败: {e}");
        sys.exit(1)

    print("正在加载交易日历...")
    try:
        cal_end = (datetime.strptime(end_date_str, '%Y%m%d') + timedelta(days=365)).strftime('%Y%m%d')
        GLOBAL_TRADE_CAL_DF = pro.trade_cal(exchange='SSE', start_date=TRAIN_DATA_FETCH_START_DATE, end_date=cal_end)
        GLOBAL_TRADE_CAL_DF.rename(columns={'cal_date': 'trade_date'}, inplace=True)
        GLOBAL_TRADE_CAL_DF['trade_date_dt'] = pd.to_datetime(GLOBAL_TRADE_CAL_DF['trade_date'])
        GLOBAL_TRADE_CAL_DF['is_open'] = pd.to_numeric(GLOBAL_TRADE_CAL_DF['is_open'])
    except Exception as e:
        print(f"错误: 获取交易日历失败: {e}");
        sys.exit(1)

    securities_to_process = [{'name': name, 'sector': cat} for cat, names in stock_info_dict.items() for name in names]

    adj_factors = None
    if USE_ADJUSTED_DATA:
        print("\n正在获取所有待处理股票的复权因子...")
        # 从 all_securities_basic 中查找 ts_code
        stock_ts_codes_to_fetch = []
        for s in securities_to_process:
            # 传递 category 以缩小查找范围
            ts_code_found, sec_type_found, _, _ = get_security_info_from_name(s['name'], all_securities_basic,
                                                                              s['sector'])
            if ts_code_found and sec_type_found == 'stock':
                stock_ts_codes_to_fetch.append(ts_code_found)

        if stock_ts_codes_to_fetch:
            try:
                adj_factors = pro.adj_factor(ts_code=','.join(filter(None, stock_ts_codes_to_fetch)),
                                             start_date=TRAIN_DATA_FETCH_START_DATE, end_date=end_date_str)
            except Exception as e:
                print(f"警告: 获取复权因子失败: {e}")

    num_cores = min(args.num_cores, len(securities_to_process))
    print(f"\n--- 开始并行处理 {len(securities_to_process)} 支证券, 使用 {num_cores} 个进程 ---")

    tasks = [(item, GLOBAL_TRADE_CAL_DF, all_securities_basic, end_date_str, adj_factors) for item in
             securities_to_process]

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(pool.starmap(_worker_fetch_data, tasks))

    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_list = [f"{r['name']} ({r['ts_code']})" for r in results if r['status'] == 'failed']
    skipped_list = [r['name'] for r in results if r['status'] == 'skipped_no_ts_code']

    print("\n--- 证券数据获取与处理完成总结 ---")
    print(f"总计: {len(results)}, 成功: {success_count}, 失败: {len(failed_list)}, 跳过: {len(skipped_list)}")
    if failed_list: print(f"  失败列表: {', '.join(failed_list)}")
    if skipped_list: print(f"  跳过列表 (未找到代码): {', '.join(skipped_list)}")
    print("脚本执行完毕。")