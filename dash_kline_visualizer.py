import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import glob
import subprocess
import sys
from datetime import datetime, timedelta
import traceback
from typing import Optional, Tuple, List, Dict, Any
import tushare as ts
import numpy as np
import ast
import re

TUSHARE_TOKEN = '5c9bcf56aeee0f2738748e413c0bd3112e22b0897618eaf7f9b4ca41'
CSV_DATA_PREDICT_BASE_DIR = 'csv_data/predict'
TRADING_METRICS_OUTPUT_BASE_BASE_DIR = 'output_filtered_signals'
BUY_COLOR = 'red'
SELL_COLOR = 'green'
PYTHON_EXECUTABLE = sys.executable
GET_STOCK_DATA_SCRIPT = "get_stock_data.py"
RUN_ALL_TRAINING_SCRIPT = "filter_trading_signals.py"

SCANNED_RESULTS_STOCK_INFO = None
LOADED_STOCK_PLOTTING_DATA: Dict[Tuple[str, str], pd.DataFrame] = {}

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

STOCK_BASIC_DF = None


def load_stock_basic_info():
    global STOCK_BASIC_DF
    if STOCK_BASIC_DF is None:
        print("Dash: 正在从Tushare获取所有证券基础信息(股票、指数、基金)...")
        try:
            stock_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry,list_status');
            stock_df['type'] = 'stock'

            markets = ['CSI', 'SSE', 'SZSE', 'CICC', 'SW', 'MSCI', 'OTH']
            index_dfs = [pro.index_basic(market=m, fields='ts_code,name') for m in markets]
            index_df = pd.concat(index_dfs, ignore_index=True).drop_duplicates(subset=['ts_code']);
            index_df['type'] = 'index'
            index_df['industry'] = '指数'

            fund_df = pro.fund_basic(market='E', fields='ts_code,name')
            fund_df['type'] = 'fund'
            fund_df['industry'] = '基金'

            STOCK_BASIC_DF = pd.concat(
                [stock_df[['ts_code', 'symbol', 'name', 'industry', 'list_status', 'type']],
                 index_df[['ts_code', 'name', 'industry', 'type']],
                 fund_df[['ts_code', 'name', 'industry', 'type']]],
                ignore_index=True
            )
            if 'list_status' not in STOCK_BASIC_DF.columns:
                STOCK_BASIC_DF['list_status'] = 'L'
            if 'symbol' not in STOCK_BASIC_DF.columns:
                STOCK_BASIC_DF['symbol'] = STOCK_BASIC_DF['ts_code']

            print(f"Dash: 所有证券基础信息获取完毕。共 {len(STOCK_BASIC_DF)} 条记录。")
        except Exception as e:
            print(f"Dash: 获取所有证券基础信息失败: {e}")
            traceback.print_exc()


def _scan_available_results_stocks_for_dropdown() -> List[Dict[str, str]]:
    available_stocks_list = []
    base_dir = TRADING_METRICS_OUTPUT_BASE_BASE_DIR

    if not os.path.exists(base_dir):
        return []

    for root, dirs, files in os.walk(base_dir):
        metrics_files = [f for f in files if f.endswith('_metrics.csv')]
        if metrics_files:
            relative_path = os.path.relpath(root, base_dir)
            path_parts = relative_path.split(os.sep)

            if len(path_parts) == 2:
                category_name_in_dir = path_parts[0]
                full_stock_name_in_dir = path_parts[1]

                display_name = full_stock_name_in_dir
                ts_code = 'N/A'

                if STOCK_BASIC_DF is not None and not STOCK_BASIC_DF.empty:
                    match = STOCK_BASIC_DF[STOCK_BASIC_DF['name'] == full_stock_name_in_dir]
                    if match.empty:
                        match = STOCK_BASIC_DF[
                            STOCK_BASIC_DF['name'].str.contains(full_stock_name_in_dir, case=False, na=False)]

                    if not match.empty:
                        best_match = match.iloc[0]
                        display_name = best_match['name']
                        ts_code = best_match['ts_code']

                available_stocks_list.append({
                    'label': f"{display_name} ({ts_code})",
                    'value': full_stock_name_in_dir,
                    'ts_code': ts_code,
                    'display_name': display_name,
                    'category_name_in_dir': category_name_in_dir,
                    'full_stock_name_in_dir': full_stock_name_in_dir,
                })

    available_stocks_list.sort(key=lambda x: x['label'])
    return available_stocks_list


def _get_scanned_results_stock_info():
    global SCANNED_RESULTS_STOCK_INFO
    if SCANNED_RESULTS_STOCK_INFO is None:
        print("Dash: Scanning output_filtered_signals for available stocks...")
        load_stock_basic_info()

        scanned_enriched = _scan_available_results_stocks_for_dropdown()
        SCANNED_RESULTS_STOCK_INFO = scanned_enriched

        print(f"Dash: Found {len(SCANNED_RESULTS_STOCK_INFO)} result directories.")
    return SCANNED_RESULTS_STOCK_INFO


def get_stock_info(stock_input: str) -> Optional[Dict[str, str]]:
    scanned_results = _get_scanned_results_stock_info()
    if not scanned_results:
        print("Dash Error: No scanned results available.")
        return None

    input_lower = stock_input.strip().lower()

    for item in scanned_results:
        if item['value'].lower() == input_lower:
            return item

    for item in scanned_results:
        if item['ts_code'].lower() != 'n/a' and item['ts_code'].lower() == input_lower:
            return item

    for item in scanned_results:
        if item['display_name'].lower() == input_lower:
            return item

    for item in scanned_results:
        if item['label'].lower() == input_lower:
            return item

    print(f"Dash Warning: Could not find scanned result info for input '{stock_input}'.")
    return None


def get_file_paths_for_kline_plot(result_item: Dict[str, str], generator_selection: str) -> Tuple[
    Optional[str], Optional[str], Optional[str]]:
    if not result_item or 'ts_code' not in result_item or 'category_name_in_dir' not in result_item or 'full_stock_name_in_dir' not in result_item:
        return None, None, "Invalid result item provided for path building."

    ts_code = result_item['ts_code']
    category_name_in_dir = result_item['category_name_in_dir']
    full_stock_name_in_dir = result_item['full_stock_name_in_dir']

    data_filename = f"{ts_code.replace('.', '_')}_{category_name_in_dir}_{full_stock_name_in_dir}_predict_data.csv"
    data_csv_path = os.path.join(CSV_DATA_PREDICT_BASE_DIR, category_name_in_dir, full_stock_name_in_dir, data_filename)

    if generator_selection == 'best_strategy':
        trading_signals_filename = "best_daily_signals.csv"
    else:
        trading_signals_filename = "best_daily_signals.csv"

    trading_signals_csv_path = os.path.join(TRADING_METRICS_OUTPUT_BASE_BASE_DIR, category_name_in_dir,
                                            full_stock_name_in_dir, trading_signals_filename)

    errors = []
    if ts_code == 'N/A' or not os.path.exists(data_csv_path):
        if ts_code == 'N/A':
            errors.append(
                f"结果目录 '{category_name_in_dir}/{full_stock_name_in_dir}' 未能关联到有效的股票代码，无法构建原始数据文件名。")
        elif not os.path.exists(data_csv_path):
            errors.append(f"原始证券数据文件 '{data_csv_path}' 不存在。\n请确认已运行 get_stock_data.py 并生成了该文件。")

    if not os.path.exists(trading_signals_csv_path): errors.append(
        f"策略交易信号文件 '{trading_signals_csv_path}' 不存在。\n请确认已运行 {RUN_ALL_TRAINING_SCRIPT} 并生成了交易指标(包括信号文件)。")

    if errors: return None, None, "\n".join(errors)
    return data_csv_path, trading_signals_csv_path, None


def _run_script_and_stream_output(script_path: str, label: str):
    process_output, current_script_dir = "", os.path.dirname(os.path.abspath(__file__))

    yield f"--- {label} 开始 ---\n"
    command = [PYTHON_EXECUTABLE, script_path]
    print(f"Executing with cwd='{current_script_dir}': {' '.join(command)}")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                   bufsize=1, universal_newlines=True, cwd=current_script_dir)

        for line in process.stdout:
            process_output += line
            yield process_output

        stdout, stderr = process.communicate()
        if stdout and not stdout.strip().startswith(process_output.strip()):
            process_output = stdout
        if stderr:
            process_output += f"\n--- {label} 错误输出 (stderr) ---\n{stderr}"

        suffix = "成功" if process.returncode == 0 else f"失败 (返回码: {process.returncode})"
        if not process_output.strip().endswith(f"--- {label} 完成: {suffix} ---"):
            process_output += f"\n--- {label} 完成: {suffix} ---"
        yield process_output
    except FileNotFoundError:
        yield f"{process_output}\n错误: 未找到脚本 '{script_path}' 或 Python 解释器 '{PYTHON_EXECUTABLE}'。"
    except Exception as e:
        yield f"{process_output}\n执行 '{label}' 时发生意外错误: {e}\n{traceback.format_exc()}"


def scan_for_buy_signals(selected_date_str: str, generator_index: str) -> List[Dict[str, str]]:
    if not selected_date_str or not SCANNED_RESULTS_STOCK_INFO:
        return []

    try:
        target_date_formatted = datetime.strptime(selected_date_str, '%Y-%m-%d').strftime('%Y%m%d')
    except ValueError:
        print(f"Invalid date format for filtering: {selected_date_str}")
        return []

    matching_stocks_with_buy_signal = []
    print(f"Checking for buy signals on {target_date_formatted} among all available stocks...")

    for item in SCANNED_RESULTS_STOCK_INFO:
        category_name_in_dir = item['category_name_in_dir']
        full_stock_name_in_dir = item['full_stock_name_in_dir']

        if generator_index == 'best_strategy':
            signals_filename = 'best_daily_signals.csv'
        else:
            signals_filename = 'best_daily_signals.csv'

        signal_filepath = os.path.join(TRADING_METRICS_OUTPUT_BASE_BASE_DIR, category_name_in_dir,
                                       full_stock_name_in_dir,
                                       signals_filename)

        if not os.path.exists(signal_filepath):
            continue

        try:
            df_signals = pd.read_csv(signal_filepath, dtype={'date': str})

            filtered_signals = df_signals[
                (df_signals['date'] == target_date_formatted) &
                (df_signals['filtered_action'] == 'Buy')
                ]

            if not filtered_signals.empty:
                matching_stocks_with_buy_signal.append(item)
        except Exception as e:
            print(f"Error processing signal file {signal_filepath} for buy signal check: {e}")
            traceback.print_exc()
            continue

    return matching_stocks_with_buy_signal


def _update_stock_info_dict_in_file(file_path: str, new_stock_name: str, new_sector_name: str) -> Tuple[bool, str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        dict_start_line_idx = -1
        dict_end_line_idx = -1
        brace_depth = 0
        in_dict_block = False

        for i, line in enumerate(lines):
            if "stock_info_dict" in line and "=" in line and "{" in line:
                dict_start_line_idx = i
                in_dict_block = True
                brace_depth += line.count('{') - line.count('}')
            elif in_dict_block:
                brace_depth += line.count('{') - line.count('}')
                if brace_depth == 0:
                    dict_end_line_idx = i
                    break

        if dict_start_line_idx == -1 or dict_end_line_idx == -1:
            return False, "错误: 无法在 get_stock_data.py 中找到 stock_info_dict 定义块或其结束括号。"

        dict_str_lines = lines[dict_start_line_idx: dict_end_line_idx + 1]
        dict_str_content = "".join(dict_str_lines).strip()

        dict_match = re.search(r'stock_info_dict\s*=\s*({.*})', dict_str_content, re.DOTALL)
        if not dict_match:
            return False, "错误: 在 get_stock_data.py 中找到 stock_info_dict 行，但无法解析字典内容。"

        dict_content_str = dict_match.group(1)

        try:
            parsed_dict = ast.literal_eval(dict_content_str)
        except (SyntaxError, ValueError) as e:
            return False, f"错误: 解析 get_stock_data.py 中的 stock_info_dict 失败 (可能格式不符合预期)。详情: {e}"

        if new_sector_name not in parsed_dict:
            parsed_dict[new_sector_name] = []

        if new_stock_name in parsed_dict[new_sector_name]:
            return False, f'股票 "{new_stock_name}" 已经在板块 "{new_sector_name}" 中。无需重复添加!'
        else:
            parsed_dict[new_sector_name].append(new_stock_name)
            parsed_dict[new_sector_name].sort()

        reconstructed_dict_lines = ["stock_info_dict = {\n"]
        for sector, stocks in sorted(parsed_dict.items()):
            stocks_str = ", ".join(f'"{stock}"' for stock in stocks)
            reconstructed_dict_lines.append(f'    "{sector}": [{stocks_str}],\n')
        reconstructed_dict_lines.append("}\n")

        new_file_content_lines = lines[:dict_start_line_idx] + reconstructed_dict_lines + lines[dict_end_line_idx + 1:]

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_file_content_lines)

        return True, f'成功添加股票 "{new_stock_name}" 到板块 "{new_sector_name}"。请点击下方更新模型!'

    except Exception as e:
        return False, f"添加股票时发生未知错误: {e}\n{traceback.format_exc()}"


try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager

    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK JP', 'Source Han Sans CN', 'Arial Unicode MS'
    ]
    found_font = None
    for font_name in chinese_fonts:
        if any(font_name.lower() in font.name.lower() for font in matplotlib.font_manager.fontManager.ttflist):
            found_font = font_name
            break
    if found_font:
        print(f"Matplotlib 字体配置: 系统支持 {found_font}")
    else:
        print("警告: 未找到任何常用中文字体。图表中的中文可能无法正常显示。")
except ImportError:
    print("Matplotlib未安装,跳过字体配置检查。")

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div(style={'font-family': 'sans-serif', 'maxWidth': '100%', 'overflowX': 'hidden'}, children=[
    html.H1("股票K线图与模型预测信号", style={'textAlign': 'center'}),
    html.P("输入股票代码或名称,查看K线图、均线、买卖信号。", style={'textAlign': 'center'}),

    html.Div(className='row', style={'display': 'flex'}, children=[
        html.Div(className='three columns', style={'padding': '20px', 'flex': '1'},
                 children=[
                     html.Label("股票代码/名称:"),
                     dcc.Input(id='stock-input-query', type='text', placeholder="例如: 韦尔股份 或 603501.SH",
                               debounce=True, style={'width': '100%'}),

                     html.Div(className='row', style={'marginTop': '10px', 'display': 'flex'},
                              children=[
                                  html.Div(className='six columns', style={'flex': '1', 'paddingRight': '5px'},
                                           children=[
                                               html.Label("按板块选择:"),
                                               dcc.Dropdown(id='sector-dropdown', options=[],
                                                            placeholder="选择板块...",
                                                            clearable=True),
                                           ]),
                                  html.Div(className='six columns', style={'flex': '1', 'paddingLeft': '5px'},
                                           children=[
                                               html.Label("选择板块内股票:"),
                                               dcc.Dropdown(id='stock-in-sector-dropdown', options=[],
                                                            placeholder="选择股票...",
                                                            clearable=True),
                                           ]),
                              ]),
                     html.Div(className='row', style={'marginTop': '10px', 'display': 'flex'},
                              children=[
                                  html.Div(className='eight columns', style={'flex': '2', 'paddingRight': '5px'},
                                           children=[
                                               html.Label("选择证券 (自动补全):"),
                                               dcc.Dropdown(id='stock-dropdown-suggestions', options=[],
                                                            placeholder="输入以搜索...",
                                                            clearable=True),
                                           ]),
                                  html.Div(className='four columns', style={'flex': '1', 'paddingLeft': '5px'},
                                           children=[
                                               dcc.Loading(
                                                   id="loading-generator-dropdown",
                                                   type="circle",
                                                   children=[
                                                       html.Label("选择策略:"),
                                                       dcc.Dropdown(id='generator-dropdown',
                                                                    options=[
                                                                        {'label': '最佳策略', 'value': 'best_strategy'},
                                                                    ],
                                                                    value='best_strategy',
                                                                    clearable=False),
                                                   ]
                                               )
                                           ]),
                              ]),
                     html.Hr(style={'marginTop': '20px', 'marginBottom': '20px'}),
                     html.H4("回测策略指标(测试集首日至今)",
                             style={'textAlign': 'center', 'fontSize': '18px', 'marginTop': '25px',
                                    'marginBottom': '15px', 'fontWeight': 'bold', 'color': '#333'}),
                     dcc.Loading(
                         id="loading-spinner-metrics",
                         type="default",
                         children=[
                             html.Div(id='trading-metrics-output',
                                      style={'marginTop': '10px', 'fontSize': '14px', 'lineHeight': '1.5'})
                         ]
                     ),

                     html.Hr(style={'marginTop': '20px', 'marginBottom': '20px'}),
                     html.H4("筛选有买入信号的股票",
                             style={'textAlign': 'center', 'fontSize': '18px', 'marginTop': '25px',
                                    'marginBottom': '15px', 'fontWeight': 'bold', 'color': '#333'}),
                     html.Div(className='row', style={'display': 'flex', 'alignItems': 'flex-end', 'gap': '10px'},
                              children=[
                                  html.Div(className='seven columns', children=[
                                      html.Label("选择日期:"),
                                      dcc.DatePickerSingle(
                                          id='buy-signal-date-picker',
                                          min_date_allowed=datetime(2013, 1, 1),
                                          max_date_allowed=datetime.now() + timedelta(days=7),
                                          initial_visible_month=datetime.now(),
                                          date=str(datetime.now().date()),
                                          display_format='YYYY-MM-DD',
                                          style={}
                                      ),
                                  ]),
                                  html.Div(className='twelve columns', children=[
                                      html.Button('筛选买入信号股票', id='filter-buy-signals-button', n_clicks=0,
                                                  style={'width': '100%', 'backgroundColor': '#007bff',
                                                         'color': 'white', 'border': 'none', 'padding': '10px 0',
                                                         'cursor': 'pointer', 'borderRadius': '5px',
                                                         'display': 'flex', 'justifyContent': 'center',
                                                         'alignItems': 'center'}),
                                  ]),
                              ]),
                     dcc.Loading(
                         id="loading-spinner-filter",
                         type="default",
                         children=[
                             html.Label("筛选结果:", style={'marginTop': '10px'}),
                             html.Div(id='filtered-stocks-count-output',
                                      style={'marginBottom': '5px', 'fontSize': '14px', 'fontWeight': 'bold'}),
                             dcc.Dropdown(
                                 id='filtered-stocks-dropdown',
                                 options=[],
                                 placeholder="选择有买入信号的证券...",
                                 clearable=True,
                                 style={'width': '100%'}
                             )
                         ]
                     ),

                     html.Label("窗口大小 (天数):", style={'marginTop': '20px'}),
                     dcc.Slider(
                         id='days-to_plot-slider',
                         min=20, max=1000, step=5, value=120,
                         marks={i: str(i) for i in [20, 100, 200, 500, 1000]}
                     ),
                     html.Label("从最新日期向前偏移天数:", style={'marginTop': '20px'}),
                     dcc.Slider(
                         id='offset-days_slider',
                         min=0, max=1000, step=10, value=0,
                         marks={i: str(i) for i in [0, 100, 200, 500, 1000]}
                     ),

                     html.Label("选择下方指标:", style={'marginTop': '20px'}),
                     dcc.Dropdown(
                         id='indicator-selector',
                         options=[
                             {'label': '无', 'value': 'None'},
                             {'label': '布林带 (主图)', 'value': 'BBANDS'},
                             {'label': 'MACD (12,26,9)', 'value': 'MACD_12_26_9'},
                             {'label': 'MACD (10,20,9)', 'value': 'MACD_10_20_9'},
                             {'label': 'KDJ (14,3,3)', 'value': 'KDJ_14_3_3'},
                             {'label': 'KDJ (9,3,3)', 'value': 'KDJ_9_3_3'},
                             {'label': 'RSI', 'value': 'RSI'},
                             {'label': 'ATR', 'value': 'ATR'},
                             {'label': 'OBV', 'value': 'OBV'},
                             {'label': 'CCI', 'value': 'CCI'},
                             {'label': 'WILLR', 'value': 'WILLR'},
                             {'label': 'ADX', 'value': 'ADX'},
                             {'label': 'MFI', 'value': 'MFI'},
                             {'label': 'A/D Line', 'value': 'AD_Line'},
                         ],
                         value=['MACD_12_26_9', 'KDJ_14_3_3'],
                         multi=True,
                         clearable=True,
                         style={'width': '100%'}
                     ),

                     html.Hr(style={'marginTop': '20px', 'marginBottom': '20px'}),
                     html.H4("添加到数据获取列表",
                             style={'textAlign': 'center', 'fontSize': '18px', 'marginTop': '25px',
                                    'marginBottom': '15px', 'fontWeight': 'bold', 'color': '#333'}),
                     html.Div(className='row', style={'display': 'flex'}, children=[
                         html.Div(className='six columns', style={'flex': '1', 'paddingRight': '5px'}, children=[
                             html.Label("证券名称 (如: 中国银行 或 沪深300ETF(沪基)):"),
                             dcc.Input(id='new-stock-name', type='text', placeholder="输入证券名称",
                                       style={'width': '100%'}),
                         ]),
                         html.Div(className='six columns', style={'flex': '1', 'paddingLeft': '5px'}, children=[
                             html.Label("所属板块 (如: 银行 或 基金(场内)):", style={'marginTop': '0px'}),
                             dcc.Input(id='new-stock-sector', type='text', placeholder="输入所属板块",
                                       style={'width': '100%'}),
                         ]),
                     ]),
                     html.Button('添加到证券数据获取列表', id='add-stock-button', n_clicks=0,
                                 style={'marginTop': '10px', 'width': '100%', 'backgroundColor': '#007bff',
                                        'color': 'white', 'border': 'none', 'padding': '10px 0',
                                        'cursor': 'pointer', 'borderRadius': '5px',
                                        'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),

                     html.Button('一键更新数据与模型(数据更新约3s/只，新增模型训练约3min/只)', id='update-data-button',
                                 n_clicks=0,
                                 style={'marginTop': '10px', 'width': '100%', 'backgroundColor': '#28a745',
                                        'color': 'white', 'fontWeight': 'bold', 'border': 'none',
                                        'padding': '12px 0', 'cursor': 'pointer', 'borderRadius': '5px',
                                        'boxShadow': '0 4px 6px rgba(0,0,0,.1)',
                                        'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),

                     dcc.Loading(
                         id="loading-spinner-update",
                         type="default",
                         children=[
                             dcc.Textarea(id='update-log-output',
                                          value="点击 '一键更新数据与模型' 开始任务。\n此操作可能需要较长时间,请耐心等待。",
                                          style={'width': '100%', 'height': '150px', 'fontSize': '12px',
                                                 'fontFamily': 'monospace', 'marginTop': '10px'},
                                          readOnly=True)
                         ]
                     ),

                     html.Div(id='status-message-display',
                              style={'marginTop': '10px', 'padding': '8px', 'borderRadius': '5px', 'display': 'none'}),
                     dcc.Store(id='message-store', data={'message': '', 'timestamp': None, 'is_error': False}),
                     dcc.Interval(id='message-interval', interval=1000, n_intervals=0, disabled=True),

                     html.Div(id='error-message-output',
                              style={'color': 'red', 'marginTop': '10px', 'wordBreak': 'break-all'})
                 ]),

        html.Div(className='nine columns', style={'flex': '3'}, children=[
            dcc.Graph(id='kline-plot-output', style={'height': '90vh'})
        ])
    ])
])


@app.callback(
    Output('stock-dropdown-suggestions', 'options'),
    Output('sector-dropdown', 'options'),
    Output('stock-dropdown-suggestions', 'value'),
    Output('generator-dropdown', 'value'),
    Input('stock-dropdown-suggestions', 'id')
)
def load_initial_options(dropdown_id):
    scanned_results_list = _get_scanned_results_stock_info()

    stock_dropdown_options = [{'label': item['label'], 'value': item['value']} for item in scanned_results_list]

    sector_dropdown_options_set = {item['category_name_in_dir'] for item in scanned_results_list}
    sector_dropdown_options = [{'label': s, 'value': s} for s in sorted(list(sector_dropdown_options_set))]

    default_stock_value = None
    default_generator_value = 'best_strategy'

    if scanned_results_list:
        default_item = scanned_results_list[0]
        default_stock_value = default_item['value']

        data_csv_path, trading_signals_csv_path, error_msg_paths = get_file_paths_for_kline_plot(
            default_item, default_generator_value
        )

        if not error_msg_paths:
            try:
                df_data_preload = pd.read_csv(data_csv_path, dtype={'date': str})
                df_signals_preload = pd.read_csv(trading_signals_csv_path, dtype={'date': str})

                df_signals_preload['trade_return_pct_shifted'] = df_signals_preload['trade_return_pct'].shift(
                    -1)

                df_data_preload['date'] = pd.to_datetime(df_data_preload['date'], format='%Y%m%d',
                                                         errors='coerce')
                df_signals_preload['date'] = pd.to_datetime(df_signals_preload['date'], format='%Y%m%d',
                                                            errors='coerce')

                ohlcv_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                core_data_cols = ohlcv_cols + ['pctChg']
                all_possible_ma_cols = [f'sma_{p}' for p in [5, 10, 15, 30, 60, 120, 240]]
                all_possible_ema_cols = [f'ema_{p}' for p in [5, 10, 60, 120, 240]]

                all_macd_cols_12_26_9 = ['macd_12_26_9', 'macds_12_26_9', 'macdh_12_26_9']
                all_macd_cols_10_20_9 = ['macd_10_20_9', 'macds_10_20_9', 'macdh_10_20_9']
                all_kdj_cols_14_3_3 = ['stochk_14_3_3', 'stochd_14_3_3', 'j_kdj_14_3_3']
                all_kdj_cols_9_3_3 = ['stochk_9_3_3', 'stochd_9_3_3', 'j_kdj_9_3_3']

                all_other_indicators = ['rsi_14', 'atr_14', 'bbu_20_2', 'bbm_20_2', 'bbl_20_2', 'obv',
                                        'ad_line',
                                        'adx_14', 'dmp_14', 'dmn_14', 'mfi_14', 'cci_14', 'willr_14']

                all_indicator_cols_to_load = (all_possible_ma_cols + all_possible_ema_cols +
                                              all_macd_cols_12_26_9 + all_macd_cols_10_20_9 +
                                              all_kdj_cols_14_3_3 + all_kdj_cols_9_3_3 +
                                              all_other_indicators)

                cols_to_load_from_data = core_data_cols + [col for col in all_indicator_cols_to_load if
                                                           col in df_data_preload.columns]

                if 'date' not in cols_to_load_from_data:
                    cols_to_load_from_data.insert(0, 'date')

                df_data_preload = df_data_preload[list(set(cols_to_load_from_data))]
                df_data_preload.dropna(subset=ohlcv_cols, inplace=True)

                df_signals_preload.dropna(subset=['date', 'filtered_action'], inplace=True)
                df_signals_to_merge = df_signals_preload[['date', 'filtered_action', 'trade_return_pct_shifted']].copy()

                df_merged_preload = pd.merge(df_data_preload,
                                             df_signals_to_merge,
                                             on='date', how='inner')
                df_merged_preload = df_merged_preload.set_index('date').sort_index()

                numeric_cols_preload = ohlcv_cols[1:] + ['pctChg'] + [col for col in all_indicator_cols_to_load
                                                                      if col in df_merged_preload.columns]
                for col in numeric_cols_preload:
                    df_merged_preload[col] = pd.to_numeric(df_merged_preload[col], errors='coerce')

                if 'filtered_action' not in df_merged_preload.columns:
                    raise ValueError("Missing 'filtered_action' column during preload.")
                df_merged_preload.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

                if not df_merged_preload.empty:
                    cache_key = (default_item['value'], default_generator_value)
                    LOADED_STOCK_PLOTTING_DATA[cache_key] = df_merged_preload
                    print(f"Dash: 预加载了默认证券 {default_item['display_name']} 的数据。")
                else:
                    print(f"Dash: 预加载 {default_item['display_name']} 导致数据为空。")

            except Exception as e:
                print(f"Dash: 预加载默认证券 {default_item['display_name']} 数据时出错: {e}")
                traceback.print_exc()

    return stock_dropdown_options, sector_dropdown_options, default_stock_value, default_generator_value


@app.callback(
    Output('stock-dropdown-suggestions', 'options', allow_duplicate=True),
    Input('stock-input-query', 'value'),
    prevent_initial_call=True
)
def update_stock_dropdown_options(query):
    if not query or not SCANNED_RESULTS_STOCK_INFO:
        return []

    matching_scanned_options = []
    query_lower = query.strip().lower()

    for item in SCANNED_RESULTS_STOCK_INFO:
        if query_lower in item['label'].lower() or \
                query_lower in item['value'].lower() or \
                (item['ts_code'] != 'N/A' and query_lower in item['ts_code'].lower()) or \
                query_lower in item['display_name'].lower():
            matching_scanned_options.append({'label': item['label'], 'value': item['value']})

    matching_scanned_options.sort(key=lambda x: x['label'])
    return matching_scanned_options[:50]


@app.callback(
    Output('stock-in-sector-dropdown', 'options'),
    Output('stock-in-sector-dropdown', 'value'),
    Input('sector-dropdown', 'value')
)
def populate_stock_in_sector_dropdown(selected_sector):
    if not selected_sector or not SCANNED_RESULTS_STOCK_INFO:
        return [], None

    options = []
    for item in SCANNED_RESULTS_STOCK_INFO:
        if item['category_name_in_dir'] == selected_sector:
            options.append({'label': item['display_name'], 'value': item['value']})

    options.sort(key=lambda x: x['label'])
    return options, None


@app.callback(
    Output('stock-dropdown-suggestions', 'value', allow_duplicate=True),
    Output('stock-input-query', 'value', allow_duplicate=True),
    Input('stock-in-sector-dropdown', 'value'),
    prevent_initial_call=True
)
def update_main_stock_selection_from_sector_dropdown(selected_full_stock_name_in_dir):
    if not selected_full_stock_name_in_dir or not SCANNED_RESULTS_STOCK_INFO:
        raise dash.exceptions.PreventUpdate

    selected_item = next(
        (item for item in SCANNED_RESULTS_STOCK_INFO if item['value'] == selected_full_stock_name_in_dir), None)

    if selected_item:
        return selected_item['value'], selected_item['display_name']

    return dash.no_update, dash.no_update


@app.callback(
    Output('kline-plot-output', 'figure'),
    Output('error-message-output', 'children'),
    Output('message-store', 'data', allow_duplicate=True),
    Input('stock-dropdown-suggestions', 'value'),
    Input('generator-dropdown', 'value'),
    Input('days-to_plot-slider', 'value'),
    Input('offset-days_slider', 'value'),
    Input('indicator-selector', 'value'),
    prevent_initial_call=True
)
def plot_kline_callback(selected_full_stock_name_in_dir, generator_selection_str, days_to_plot, offset_days,
                        selected_indicator_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    plot_message_data = {'message': '', 'timestamp': None, 'is_error': False}
    error_message_output = ""

    if not selected_full_stock_name_in_dir or not SCANNED_RESULTS_STOCK_INFO:
        error_message_output = "请选择或输入证券。"
        return go.Figure(), error_message_output, plot_message_data

    selected_item = next(
        (item for item in SCANNED_RESULTS_STOCK_INFO if item['value'] == selected_full_stock_name_in_dir), None)

    if not selected_item:
        error_message_output = f"内部错误: 未找到值 '{selected_full_stock_name_in_dir}' 对应的结果数据。"
        return go.Figure(), error_message_output, plot_message_data

    ts_code = selected_item['ts_code']
    stock_name_display = selected_item['display_name']
    category_name_in_dir = selected_item['category_name_in_dir']
    full_stock_name_in_dir = selected_item['full_stock_name_in_dir']

    data_csv_path, trading_signals_csv_path, error_msg = get_file_paths_for_kline_plot(
        selected_item, generator_selection_str
    )

    if error_msg:
        error_message_output = error_msg
        return go.Figure(), error_message_output, plot_message_data

    ohlcv_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    core_data_cols = ohlcv_cols + ['pctChg']
    all_possible_ma_cols = [f'sma_{p}' for p in [5, 10, 15, 30, 60, 120, 240]]
    all_possible_ema_cols = [f'ema_{p}' for p in [5, 10, 60, 120, 240]]

    all_macd_cols_12_26_9 = ['macd_12_26_9', 'macds_12_26_9', 'macdh_12_26_9']
    all_macd_cols_10_20_9 = ['macd_10_20_9', 'macds_10_20_9', 'macdh_10_20_9']
    all_kdj_cols_14_3_3 = ['stochk_14_3_3', 'stochd_14_3_3', 'j_kdj_14_3_3']
    all_kdj_cols_9_3_3 = ['stochk_9_3_3', 'stochd_9_3_3', 'j_kdj_9_3_3']

    all_other_indicators = ['rsi_14', 'atr_14', 'bbu_20_2', 'bbm_20_2', 'bbl_20_2', 'obv', 'ad_line',
                            'adx_14', 'dmp_14', 'dmn_14', 'mfi_14', 'cci_14', 'willr_14']

    all_indicator_cols_to_load = (all_possible_ma_cols + all_possible_ema_cols +
                                  all_macd_cols_12_26_9 + all_macd_cols_10_20_9 +
                                  all_kdj_cols_14_3_3 + all_kdj_cols_9_3_3 +
                                  all_other_indicators)

    cache_key = (full_stock_name_in_dir, generator_selection_str)

    df_merged = None
    if cache_key in LOADED_STOCK_PLOTTING_DATA:
        df_merged = LOADED_STOCK_PLOTTING_DATA[cache_key]
        print(
            f"Dash: 从缓存中获取数据，用于 {stock_name_display} ({full_stock_name_in_dir}) ({generator_selection_str})。")
    else:
        try:
            df_data = pd.read_csv(data_csv_path, dtype={'date': str})
            df_signals = pd.read_csv(trading_signals_csv_path, dtype={'date': str})

            df_signals['trade_return_pct_shifted'] = df_signals['trade_return_pct'].shift(-1)

            df_data['date'] = pd.to_datetime(df_data['date'], format='%Y%m%d', errors='coerce')
            df_signals['date'] = pd.to_datetime(df_signals['date'], format='%Y%m%d', errors='coerce')

            cols_to_load_from_data = core_data_cols + [col for col in all_indicator_cols_to_load if
                                                       col in df_data.columns]

            if 'date' not in cols_to_load_from_data:
                cols_to_load_from_data.insert(0, 'date')

            df_data = df_data[list(set(cols_to_load_from_data))]
            df_data.dropna(subset=ohlcv_cols, inplace=True)

            df_signals.dropna(subset=['date', 'filtered_action'], inplace=True)
            df_signals_to_merge = df_signals[['date', 'filtered_action', 'trade_return_pct_shifted']].copy()

            df_merged = pd.merge(df_data, df_signals_to_merge, on='date', how='inner')
            df_merged = df_merged.set_index('date').sort_index()

            numeric_cols = ohlcv_cols[1:] + ['pctChg'] + [col for col in all_indicator_cols_to_load if
                                                          col in df_merged.columns]
            for col in numeric_cols: df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

            if 'filtered_action' not in df_merged.columns: raise ValueError("Missing 'filtered_action' column.")
            df_merged.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

            if df_merged.empty:
                error_message_output = "合并数据后为空。"
                return go.Figure(), error_message_output, plot_message_data

            LOADED_STOCK_PLOTTING_DATA[cache_key] = df_merged
            print(f"Dash: 加载并缓存了 {stock_name_display} ({full_stock_name_in_dir}) 的数据。")

        except Exception as e:
            print(f"Plotly 绘图时发生错误: {e}")
            traceback.print_exc()
            error_message_output = f"绘图失败: {e}"
            plot_message_data = {'message': '', 'timestamp': None, 'is_error': False}
            return go.Figure(), error_message_output, plot_message_data

    full_data_length = len(df_merged)
    end_idx = full_data_length - offset_days
    start_idx = max(0, end_idx - days_to_plot)
    df_plot = df_merged.iloc[start_idx:end_idx].copy()

    if df_plot.empty:
        error_message_output = "日期窗口内无数据。"
        return go.Figure(), error_message_output, plot_message_data

    title_gen_str = '最佳策略'

    num_rows = 2
    row_heights = [0.6, 0.15]

    INDICATOR_PLOT_CONFIGS = {
        'MACD_12_26_9': {'height_ratio': 0.125, 'title': 'MACD (12,26,9)',
                         'lines': {'DIF': 'macd_12_26_9', 'DEA': 'macds_12_26_9'},
                         'bars': {'MACDH': 'macdh_12_26_9'}, 'line_colors': {'DIF': 'blue', 'DEA': 'orange'},
                         'bar_colors': {'MACDH': [BUY_COLOR, SELL_COLOR]}, 'hlines': [0]},
        'MACD_10_20_9': {'height_ratio': 0.125, 'title': 'MACD (10,20,9)',
                         'lines': {'DIF': 'macd_10_20_9', 'DEA': 'macds_10_20_9'},
                         'bars': {'MACDH': 'macdh_10_20_9'}, 'line_colors': {'DIF': 'purple', 'DEA': 'cyan'},
                         'bar_colors': {'MACDH': [BUY_COLOR, SELL_COLOR]}, 'hlines': [0]},
        'KDJ_14_3_3': {'height_ratio': 0.125, 'title': 'KDJ (14,3,3)',
                       'lines': {'K': 'stochk_14_3_3', 'D': 'stochd_14_3_3', 'J': 'j_kdj_14_3_3'},
                       'line_colors': {'K': 'purple', 'D': 'cyan', 'J': 'magenta'}, 'hlines': [20, 80]},
        'KDJ_9_3_3': {'height_ratio': 0.125, 'title': 'KDJ (9,3,3)',
                      'lines': {'K': 'stochk_9_3_3', 'D': 'stochd_9_3_3', 'J': 'j_kdj_9_3_3'},
                      'line_colors': {'K': '#FF69B4', 'D': '#00BFFF', 'J': '#FF4500'}, 'hlines': [20, 80]},
        'RSI': {'height_ratio': 0.08, 'title': 'RSI', 'lines': {'RSI': 'rsi_14'}, 'line_colors': {'RSI': 'green'},
                'hlines': [30, 70]},
        'ATR': {'height_ratio': 0.08, 'title': 'ATR', 'lines': {'ATR': 'atr_14'}, 'line_colors': {'ATR': 'purple'}},
        'OBV': {'height_ratio': 0.08, 'title': 'OBV', 'lines': {'OBV': 'obv'}, 'line_colors': {'OBV': 'blue'}},
        'CCI': {'height_ratio': 0.08, 'title': 'CCI', 'lines': {'CCI': 'cci_14'}, 'line_colors': {'CCI': 'darkred'},
                'hlines': [100, -100]},
        'WILLR': {'height_ratio': 0.08, 'title': 'WILLR', 'lines': {'WILLR': 'willr_14'},
                  'line_colors': {'WILLR': 'goldenrod'}, 'hlines': [-20, -80]},
        'ADX': {'height_ratio': 0.08, 'title': 'ADX', 'lines': {'ADX': 'adx_14', '+DI': 'dmp_14', '-DI': 'dmn_14'},
                'line_colors': {'ADX': 'blue', '+DI': BUY_COLOR, '-DI': SELL_COLOR}},
        'MFI': {'height_ratio': 0.08, 'title': 'MFI', 'lines': {'MFI': 'mfi_14'},
                'line_colors': {'MFI': 'chocolate'}, 'hlines': [20, 80]},
        'AD_Line': {'height_ratio': 0.08, 'title': 'A/D Line', 'lines': {'A/D Line': 'ad_line'},
                    'line_colors': {'A/D Line': 'teal'}},
    }

    indicators_to_plot_in_rows = []
    if selected_indicator_list:
        for ind_name in selected_indicator_list:
            if ind_name == 'None' or ind_name == 'BBANDS':
                continue
            if ind_name in INDICATOR_PLOT_CONFIGS:
                data_available = True
                if 'lines' in INDICATOR_PLOT_CONFIGS[ind_name]:
                    for line_col in INDICATOR_PLOT_CONFIGS[ind_name]['lines'].values():
                        if line_col not in df_plot.columns or df_plot[line_col].isnull().all():
                            data_available = False
                            break
                if data_available and 'bars' in INDICATOR_PLOT_CONFIGS[ind_name]:
                    for bar_col in INDICATOR_PLOT_CONFIGS[ind_name]['bars'].values():
                        if bar_col not in df_plot.columns or df_plot[bar_col].isnull().all():
                            data_available = False
                            break

                if data_available:
                    indicators_to_plot_in_rows.append(ind_name)
                    num_rows += 1
                    row_heights.append(INDICATOR_PLOT_CONFIGS[ind_name]['height_ratio'])
                else:
                    print(f"Warning: Data for {ind_name} is missing or all NaN. Skipping plot for {ind_name}.")

    total_height_sum = sum(row_heights)
    if total_height_sum > 0:
        row_heights = [h / total_height_sum for h in row_heights]

    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=row_heights)

    first_date_year = df_plot.index.min().year
    last_date_year = df_plot.index.max().year

    if first_date_year != last_date_year:
        date_format = "%Y-%m-%d"
    else:
        date_format = "%m-%d"

    df_plot['Date_Str_Full'] = df_plot.index.strftime("%Y-%m-%d")
    df_plot['Date_Str_Hover'] = df_plot.index.strftime("%Y-%m-%d")

    full_raw_dates_in_plot = df_plot['Date_Str_Full'].tolist()
    full_formatted_dates_in_plot = df_plot.index.strftime(date_format).tolist()

    final_tick_vals = []
    final_tick_texts = []

    total_points_in_plot = len(full_raw_dates_in_plot)
    if total_points_in_plot > 0:
        final_tick_vals.append(full_raw_dates_in_plot[0])
        final_tick_texts.append(full_formatted_dates_in_plot[0])

        if total_points_in_plot > 1:
            final_tick_vals.append(full_raw_dates_in_plot[-1])
            final_tick_texts.append(full_formatted_dates_in_plot[-1])

        desired_intermediate_ticks = min(8, total_points_in_plot - 2)
        if desired_intermediate_ticks > 0:
            step_size = max(1, (total_points_in_plot - 1) // (desired_intermediate_ticks + 1))
            for i in range(1, total_points_in_plot - 1):
                if i % step_size == 0:
                    if full_raw_dates_in_plot[i] not in final_tick_vals:
                        final_tick_vals.append(full_raw_dates_in_plot[i])
                        final_tick_texts.append(full_formatted_dates_in_plot[i])

            combined_ticks = sorted(list(zip(final_tick_vals, final_tick_texts)),
                                    key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))
            final_tick_vals = [item[0] for item in combined_ticks]
            final_tick_texts = [item[1] for item in combined_ticks]

    hover_texts = []
    for idx, row in df_plot.iterrows():
        lines = [
            f'<b>日期</b>: {row["Date_Str_Hover"]}',
            f'<b>开盘</b>: {row["open"]:.2f}',
            f'<b>最高</b>: {row["high"]:.2f}',
            f'<b>最低</b>: {row["low"]:.2f}',
            f'<b>收盘</b>: {row["close"]:.2f}'
        ]

        if 'pctChg' in row and pd.notna(row['pctChg']):
            lines.append(f'<b>涨跌幅</b>: {row["pctChg"]:.2f}%')

        for p in all_possible_ma_cols:
            if p in df_plot.columns and pd.notna(row[p]):
                lines.append(f'<b>{p.upper().replace("SMA_", "MA")}</b>: {row[p]:.2f}')

        if row['filtered_action'] == 'Buy':
            lines.append('<b>操作</b>: 买入')
        elif row['filtered_action'] == 'Sell':
            lines.append(f'<b>操作</b>: 卖出')
            if 'trade_return_pct_shifted' in row and pd.notna(row['trade_return_pct_shifted']):
                lines.append(f'<b>收益率</b>: {row["trade_return_pct_shifted"]:.2f}%')

        hover_texts.append('<br>'.join(lines))

    fig.add_trace(go.Candlestick(
        x=df_plot['Date_Str_Full'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'],
        close=df_plot['close'],
        name='K线图', increasing_line_color=BUY_COLOR, decreasing_line_color=SELL_COLOR,
        hoverinfo='text', hovertext=hover_texts,
    ), row=1, col=1)

    ma_colors = ['#FFD700', '#FFA500', '#8A2BE2', '#00BFFF', '#ADFF2F', '#FF6347']
    for i, period in enumerate([5, 10, 30, 60, 120, 240]):
        ma_col_name_in_df = f'sma_{period}'
        if ma_col_name_in_df in df_plot.columns and not df_plot[
            ma_col_name_in_df].isnull().all():
            fig.add_trace(go.Scatter(
                x=df_plot['Date_Str_Full'], y=df_plot[ma_col_name_in_df], mode='lines', name=f'MA{period}',
                line=dict(color=ma_colors[i % len(ma_colors)], width=1),
                hoverinfo='none',
                showlegend=True
            ), row=1, col=1)

    if selected_indicator_list and 'BBANDS' in selected_indicator_list:
        bbu_col = 'bbu_20_2'
        bbm_col = 'bbm_20_2'
        bbl_col = 'bbl_20_2'

        if all(col in df_plot.columns for col in [bbu_col, bbm_col, bbl_col]):
            print("Plotting Bollinger Bands on main chart...")
            fig.add_trace(go.Scatter(
                x=df_plot['Date_Str_Full'], y=df_plot[bbl_col], mode='lines',
                line=dict(width=1.5, color='purple'),
                name='BB-下轨', hoverinfo='none', showlegend=True
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_plot['Date_Str_Full'], y=df_plot[bbu_col], mode='lines',
                line=dict(width=1.5, color='purple'),
                fill='tonexty',
                fillcolor='rgba(128, 0, 128, 0.1)',
                name='BB-上轨', hoverinfo='none', showlegend=True
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_plot['Date_Str_Full'], y=df_plot[bbm_col], mode='lines',
                line=dict(width=1.5, dash='dash', color='orange'),
                name='BB-中轨', hoverinfo='none', showlegend=True
            ), row=1, col=1)
        else:
            print(f"Warning: Bollinger Band columns ({bbu_col}, {bbm_col}, {bbl_col}) not found. Skipping.")

    buy_signals_df_filtered = df_plot[df_plot['filtered_action'] == 'Buy'].copy()
    if not buy_signals_df_filtered.empty:
        buy_signal_y = buy_signals_df_filtered['low'] * 0.99
        fig.add_trace(go.Scatter(
            x=buy_signals_df_filtered['Date_Str_Full'],
            y=buy_signal_y, mode='markers',
            marker=dict(symbol='triangle-up', size=10, color=BUY_COLOR),
            name='买入信号', hoverinfo='none',
        ), row=1, col=1)

    sell_signals_df_filtered = df_plot[df_plot['filtered_action'] == 'Sell'].copy()
    if not sell_signals_df_filtered.empty:
        sell_signal_y = df_plot.loc[sell_signals_df_filtered.index, 'high'] * 1.01
        fig.add_trace(go.Scatter(
            x=df_plot.loc[sell_signals_df_filtered.index, 'Date_Str_Full'],
            y=sell_signal_y, mode='markers',
            marker=dict(symbol='triangle-down', size=10, color=SELL_COLOR),
            name='卖出信号', hoverinfo='none',
        ), row=1, col=1)

        for idx, row_signal in sell_signals_df_filtered.iterrows():
            return_val = row_signal.get('trade_return_pct_shifted', None)

            if pd.notna(return_val):
                return_val_str = f"{return_val:.2f}%"
                annotation_text_color = BUY_COLOR if return_val >= 0 else SELL_COLOR
                annotation_bg_color = 'rgba(255, 0, 0, 0.2)' if return_val >= 0 else 'rgba(0, 128, 0, 0.2)'
                annotation_border_color = BUY_COLOR if return_val >= 0 else SELL_COLOR
                text_y_position = row_signal['high'] * 1.025

                fig.add_annotation(x=row_signal['Date_Str_Full'], y=text_y_position, text=return_val_str,
                                   showarrow=False,
                                   font=dict(size=9, color=annotation_text_color, family='Arial, sans-serif'),
                                   bgcolor=annotation_bg_color, bordercolor=annotation_border_color, borderwidth=1,
                                   borderpad=2,
                                   yanchor="bottom", row=1, col=1)

    fig.add_trace(go.Bar(
        x=df_plot['Date_Str_Full'], y=df_plot['volume'], name='成交量',
        marker_color=np.where(df_plot['close'] > df_plot['open'], BUY_COLOR, SELL_COLOR),
        hoverinfo='x+y', showlegend=False
    ), row=2, col=1)

    current_plot_row = 3

    for ind_to_plot in indicators_to_plot_in_rows:
        config = INDICATOR_PLOT_CONFIGS[ind_to_plot]

        if 'lines' in config:
            for line_name, col_name_actual in config['lines'].items():
                if col_name_actual in df_plot.columns and not df_plot[col_name_actual].isnull().all():
                    fig.add_trace(go.Scatter(x=df_plot['Date_Str_Full'], y=df_plot[col_name_actual], mode='lines',
                                             name=line_name,
                                             line=dict(color=config['line_colors'][line_name], width=1),
                                             showlegend=False), row=current_plot_row, col=1)

        if 'bars' in config:
            for bar_name, col_name_actual in config['bars'].items():
                if col_name_actual in df_plot.columns and not df_plot[col_name_actual].isnull().all():
                    bar_colors = config['bar_colors'][bar_name]
                    bar_marker_colors = np.where(df_plot[col_name_actual] >= 0, bar_colors[0], bar_colors[1])
                    fig.add_trace(go.Bar(x=df_plot['Date_Str_Full'], y=df_plot[col_name_actual], name=bar_name,
                                         marker_color=bar_marker_colors, showlegend=False), row=current_plot_row,
                                  col=1)

        if 'hlines' in config:
            for hval in config['hlines']:
                fig.add_hline(y=hval, line_dash="dot", line_color="gray", row=current_plot_row, col=1)

        fig.update_yaxes(title_text='', row=current_plot_row, col=1, title_font=dict(size=12), side='left',
                         fixedrange=False, showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.add_annotation(xref="x domain", yref="y domain", x=0.01, y=0.98, text=config['title'], showarrow=False,
                           font=dict(size=12, color="black"), xanchor='left', yanchor='top', row=current_plot_row,
                           col=1)

        current_plot_row += 1

    xaxis_configs = [
        dict(type='category', rangeslider=dict(visible=False), tickvals=final_tick_vals, ticktext=final_tick_texts,
             tickangle=-45)]
    for i in range(1, num_rows):
        xaxis_configs.append(
            dict(type='category', tickvals=final_tick_vals, ticktext=final_tick_texts, tickangle=-45))

    fig.update_layout(
        title_text=f"{stock_name_display} ({ts_code}) - K线图 ({title_gen_str} 策略信号)",
        title_x=0.5,
        **{f'xaxis{i + 1}': config for i, config in enumerate(xaxis_configs)},
        height=600 + (len(indicators_to_plot_in_rows) * 100),
        hovermode='x unified', template='plotly_white',
        margin=dict(l=50, r=20, t=60, b=50),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                    bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.5)'),
    )
    fig.update_yaxes(title_text='', row=1, col=1, title_font=dict(size=12), fixedrange=False,
                     showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(title_text='', row=2, col=1, title_font=dict(size=12), side='left', fixedrange=False,
                     showline=True, linewidth=1, linecolor='black', mirror=True)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.01, y=0.98, text="K线图", showarrow=False,
        font=dict(size=12, color="black"), xanchor='left', yanchor='top', row=1, col=1
    )
    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.01, y=0.98, text="成交量", showarrow=False,
        font=dict(size=12, color="black"), xanchor='left', yanchor='top', row=2, col=1
    )

    plot_message_data = {'message': f"K线图生成成功! {stock_name_display} ({ts_code})",
                         'timestamp': str(datetime.now()), 'is_error': False}

    return fig, "", plot_message_data


@app.callback(
    Output('trading-metrics-output', 'children'),
    Input('stock-dropdown-suggestions', 'value'),
    Input('generator-dropdown', 'value'),
    prevent_initial_call=False
)
def load_trading_metrics(selected_full_stock_name_in_dir, generator_selection_str):
    if not selected_full_stock_name_in_dir or not generator_selection_str or not SCANNED_RESULTS_STOCK_INFO:
        return html.Div("等待选择证券,加载回测指标...", style={'color': 'gray'})

    selected_item = next(
        (item for item in SCANNED_RESULTS_STOCK_INFO if item['value'] == selected_full_stock_name_in_dir), None)

    if not selected_item:
        return html.Div("加载指标失败: 未找到对应的结果数据。", style={'color': 'red'})

    ts_code = selected_item['ts_code']
    stock_name_display = selected_item['display_name']
    category_name_in_dir = selected_item['category_name_in_dir']
    full_stock_name_in_dir = selected_item['full_stock_name_in_dir']

    if generator_selection_str == 'best_strategy':
        metrics_filename = "best_metrics.csv"
    else:
        return html.Div("请选择 '最佳策略' 查看指标。", style={'color': 'orange'})

    metrics_csv_path = os.path.join(TRADING_METRICS_OUTPUT_BASE_BASE_DIR, category_name_in_dir, full_stock_name_in_dir,
                                    metrics_filename)

    if os.path.exists(metrics_csv_path):
        try:
            df_metrics = pd.read_csv(metrics_csv_path)
            if not df_metrics.empty:
                num_trades = df_metrics['num_trades'].iloc[0]
                cumulative_return_percentage = df_metrics['cumulative_return_percentage'].iloc[0]
                avg_return_per_trade_percentage = df_metrics['avg_return_per_trade_percentage'].iloc[0]
                win_rate_percentage = df_metrics['win_rate_percentage'].iloc[0]
                profit_loss_ratio = df_metrics['profit_loss_ratio'].iloc[0]
                max_single_trade_drawdown_percentage = df_metrics['max_single_trade_drawdown_percentage'].iloc[0]

                return html.Div([
                    html.P(f"完成交易笔数: {int(num_trades)}"),
                    html.P(f"累积收益率: {cumulative_return_percentage:.2f}%"),
                    html.P(f"平均每笔收益率: {avg_return_per_trade_percentage:.2f}%"),
                    html.P(f"胜率: {win_rate_percentage:.2f}%"),
                    html.P(f"盈亏比: {profit_loss_ratio:.2f}"),
                    html.P(f"单笔最大回撤: {max_single_trade_drawdown_percentage:.2f}%"),
                ])
            else:
                return html.Div("策略指标文件为空,请检查。", style={'color': 'orange'})
        except Exception as e:
            print(f"Error reading metrics file {metrics_csv_path}: {e}")
            traceback.print_exc()
            return html.Div(f"读取策略指标文件失败: {e}", style={'color': 'red'})
    else:
        return html.Div(f"未找到策略指标文件: {metrics_filename}。", style={'color': 'orange'})


@app.callback(
    Output('status-message-display', 'children'),
    Output('status-message-display', 'style'),
    Output('message-interval', 'disabled'),
    Output('message-interval', 'n_intervals'),
    Input('message-store', 'data'),
    Input('message-interval', 'n_intervals'),
    State('status-message-display', 'style')
)
def update_status_message(stored_data, n_intervals, current_style):
    message = stored_data.get('message', '')
    timestamp_str = stored_data.get('timestamp')
    is_error = stored_data.get('is_error', False)

    display_style = {'display': 'none'}
    interval_disabled = True
    reset_n_intervals = 0

    if message:
        bg_color = '#d4edda' if not is_error else '#f8d7da'
        text_color = '#155724' if not is_error else '#721c24'
        border_color = '#c3e6cb' if not is_error else '#f5c6cb'

        display_style = {
            'marginTop': '10px',
            'padding': '8px',
            'borderRadius': '5px',
            'display': 'block',
            'backgroundColor': bg_color,
            'color': text_color,
            'border': f'1px solid {border_color}'
        }

        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
            time_since_message = (datetime.now() - timestamp).total_seconds()
            if time_since_message >= 3:
                message = ''
                display_style['display'] = 'none'
                interval_disabled = True
                reset_n_intervals = 0
            else:
                interval_disabled = False
                reset_n_intervals = 0
        else:
            interval_disabled = True
            reset_n_intervals = 0

    return message, display_style, interval_disabled, reset_n_intervals


@app.callback(
    Output('update-log-output', 'value'),
    Output('error-message-output', 'children', allow_duplicate=True),
    Input('update-data-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_data_and_models_callback(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    log_output_text = ""
    error_message = ""

    def _gen_log_stream(script_path, label):
        nonlocal log_output_text
        yield f"--- {label} 开始 ---\n"
        full_log = ""
        try:
            process = subprocess.Popen([PYTHON_EXECUTABLE, script_path], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding='utf-8', bufsize=1, universal_newlines=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__)))
            for line in process.stdout:
                full_log += line
                yield full_log
            process.wait()
            ret_code = process.returncode
            full_log += f"\n--- {label} 完成: {'成功' + (' (需要重启Dash以应用股票列表变更)' if 'stock_info_dict' in full_log else '') if ret_code == 0 else f'失败 (返回码: {ret_code})'} ---\n"
            log_output_text = full_log
            yield full_log
            return ret_code == 0
        except Exception as e:
            full_log += f"\n执行 '{label}' 时发生意外错误: {e}\n{traceback.format_exc()}"
            log_output_text = full_log
            yield full_log
            return False

    get_data_success = False
    for chunk in _gen_log_stream(GET_STOCK_DATA_SCRIPT, "数据获取"):
        log_output_text = chunk
    if f"--- 数据获取 完成: 成功 ---" in log_output_text:
        get_data_success = True

    if not get_data_success:
        error_message = "数据获取失败,任务终止。"
        return log_output_text, error_message

    run_training_success = False
    for chunk in _gen_log_stream(RUN_ALL_TRAINING_SCRIPT, "信号过滤与策略评估"):
        log_output_text = chunk
    if f"--- 信号过滤与策略评估 完成: 成功 ---" in log_output_text:
        run_training_success = True

    if not run_training_success:
        error_message = "信号过滤/策略评估失败。"
        return log_output_text, error_message

    return log_output_text, "所有数据和模型更新任务已尝试执行完毕!"


@app.callback(
    Output('filtered-stocks-dropdown', 'options'),
    Output('filtered-stocks-dropdown', 'value'),
    Output('filtered-stocks-count-output', 'children'),
    Input('filter-buy-signals-button', 'n_clicks'),
    State('buy-signal-date-picker', 'date'),
    State('generator-dropdown', 'value'),
    prevent_initial_call=True
)
def filter_buy_signals_callback(n_clicks, selected_date, generator_index):
    if n_clicks == 0 or not selected_date:
        raise dash.exceptions.PreventUpdate

    if not generator_index:
        return [], None, "请选择一个策略来筛选。"

    matching_stocks = scan_for_buy_signals(selected_date, generator_index)

    options = []
    default_value = None
    count_message = ""

    num_matching = len(matching_stocks)

    if num_matching == 0:
        options.append({'label': f"在 {selected_date} 没有找到有买入信号的证券。", 'value': '', 'disabled': True})
        count_message = f"找到 0 支有买入信号的证券。"
    else:
        options = [{'label': item['label'], 'value': item['value']} for item in matching_stocks]

        if options:
            default_value = options[0]['value']
        count_message = f"找到 {num_matching} 支有买入信号的证券。"

    return options, default_value, count_message


@app.callback(
    Output('stock-dropdown-suggestions', 'value', allow_duplicate=True),
    Output('stock-input-query', 'value', allow_duplicate=True),
    Input('filtered-stocks-dropdown', 'value'),
    prevent_initial_call=True
)
def link_filtered_stock_to_main_selection(selected_full_stock_name_in_dir_from_filter):
    if not selected_full_stock_name_in_dir_from_filter or not SCANNED_RESULTS_STOCK_INFO:
        raise dash.exceptions.PreventUpdate

    selected_item = next(
        (item for item in SCANNED_RESULTS_STOCK_INFO if item['value'] == selected_full_stock_name_in_dir_from_filter),
        None)

    if selected_item:
        return selected_item['value'], selected_item['display_name']

    return dash.no_update, dash.no_update


@app.callback(
    Output('message-store', 'data', allow_duplicate=True),
    Input('add-stock-button', 'n_clicks'),
    State('new-stock-name', 'value'),
    State('new-stock-sector', 'value'),
    prevent_initial_call=True
)
def add_stock_to_list(n_clicks, stock_name, sector_name):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    if not stock_name or not sector_name:
        return {'message': "证券名称和所属板块不能为空。", 'timestamp': str(datetime.now()), 'is_error': True}

    get_stock_data_path = os.path.join(os.path.dirname(__file__), GET_STOCK_DATA_SCRIPT)

    success, message = _update_stock_info_dict_in_file(get_stock_data_path, stock_name, sector_name)

    return {'message': message, 'timestamp': str(datetime.now()), 'is_error': not success}


if __name__ == '__main__':
    print("Starting Dash application...")
    _get_scanned_results_stock_info()

    app.run(debug=True, use_reloader=False)