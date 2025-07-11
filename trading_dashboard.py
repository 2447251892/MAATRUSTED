#trading_dashboard.py

import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import torch
import os
import glob
import numpy as np
import traceback

EXPERIMENT_BASE_DIR = 'output_filtered_signals'
RAW_DATA_BASE_DIR = 'csv_data/predict'

def find_available_experiments():
    exp_info = {}
    if not os.path.exists(EXPERIMENT_BASE_DIR):
        print(f"警告：实验目录 '{EXPERIMENT_BASE_DIR}' 不存在。")
        return exp_info
    for sector in os.listdir(EXPERIMENT_BASE_DIR):
        sector_path = os.path.join(EXPERIMENT_BASE_DIR, sector)
        if not os.path.isdir(sector_path): continue
        exp_info[sector] = {}
        for stock_name in os.listdir(sector_path):
            stock_path = os.path.join(sector_path, stock_name)
            if os.path.isdir(stock_path) and any(f.endswith('_daily_signals.csv') for f in os.listdir(stock_path)):
                exp_info[sector][stock_name] = stock_path
    return exp_info

def load_kline_and_signals(stock_name, sector_name, experiment_dir, model_signal_file):
    data_path_pattern = os.path.join(RAW_DATA_BASE_DIR, sector_name, stock_name, "*.csv")
    files = glob.glob(data_path_pattern)
    if not files:
        raise FileNotFoundError(f"找不到股票 {stock_name} 的K线数据文件。")
    cols_to_load = ['date', 'open', 'high', 'low', 'close', 'volume']
    df_kline = pd.read_csv(files[0], usecols=lambda c: c in cols_to_load, dtype={'date': str}) # Read date as string first

    signal_filepath = os.path.join(experiment_dir, model_signal_file)
    if not os.path.exists(signal_filepath):
        raise FileNotFoundError(f"找不到信号文件: {signal_filepath}")
    df_signals = pd.read_csv(signal_filepath, dtype={'date': str}) # Read date as string first

    # --- MODIFICATION START ---
    # Use the correct date format '%Y-%m-%d'
    df_kline['date'] = pd.to_datetime(df_kline['date'], format='%Y%m%d', errors='coerce') # Original data is %Y%m%d
    df_signals['date'] = pd.to_datetime(df_signals['date'], format='%Y-%m-%d', errors='coerce') # Signal file is %Y-%m-%d
    # --- MODIFICATION END ---

    df_kline.dropna(subset=['date'], inplace=True)
    df_signals.dropna(subset=['date'], inplace=True)

    df_merged = pd.merge(df_kline, df_signals, on='date', how='inner')
    df_merged.sort_values('date', inplace=True)

    return df_merged

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/codepen/bWLwgP.css'])

AVAILABLE_EXPERIMENTS = find_available_experiments()
SECTOR_OPTIONS = [{'label': s, 'value': s} for s in sorted(AVAILABLE_EXPERIMENTS.keys())]

app.layout = html.Div([
    html.H1("交易信号可视化仪表盘", style={'textAlign': 'center'}),
    html.Div(className='row', style={'padding': '20px'}, children=[
        html.Div(className='four columns', children=[
            html.Label("选择板块:"),
            dcc.Dropdown(id='sector-dropdown', options=SECTOR_OPTIONS),
            html.Label("选择股票:"),
            dcc.Dropdown(id='stock-dropdown'),
            html.Label("选择模型/策略:"),
            dcc.Dropdown(id='model-dropdown'),
        ]),
        html.Div(className='eight columns', children=[
            dcc.Loading(dcc.Graph(id='kline-plot', style={'height': '80vh'}))
        ]),
    ]),
    html.Div(id='error-output', style={'color': 'red', 'padding': '20px', 'whiteSpace': 'pre-wrap'})
])

@app.callback(Output('stock-dropdown', 'options'), Input('sector-dropdown', 'value'))
def update_stock_options(selected_sector):
    if not selected_sector: return []
    stocks = AVAILABLE_EXPERIMENTS.get(selected_sector, {})
    return [{'label': stock, 'value': stock} for stock in sorted(stocks.keys())]

@app.callback(
    Output('model-dropdown', 'options'),
    Input('stock-dropdown', 'value'),
    State('sector-dropdown', 'value')
)
def update_model_options(selected_stock, selected_sector):
    if not selected_stock or not selected_sector: return []
    experiment_dir = AVAILABLE_EXPERIMENTS.get(selected_sector, {}).get(selected_stock)
    if not experiment_dir: return []
    models_found = []
    for f in os.listdir(experiment_dir):
        if f.endswith('_daily_signals.csv'):
            parts = f.replace('_daily_signals.csv', '').split('_')
            label = f"{parts[0]} ({parts[1]})"
            value = f
            models_found.append({'label': label, 'value': value})
    return sorted(models_found, key=lambda x: x['label'])

@app.callback(
    Output('kline-plot', 'figure'),
    Output('error-output', 'children'),
    Input('model-dropdown', 'value'),
    State('stock-dropdown', 'value'),
    State('sector-dropdown', 'value')
)
def update_graph(selected_signal_file, selected_stock, selected_sector):
    if not all([selected_signal_file, selected_stock, selected_sector]):
        return go.Figure(), "请选择板块、股票和模型。"
    try:
        experiment_dir = AVAILABLE_EXPERIMENTS[selected_sector][selected_stock]
        df_plot = load_kline_and_signals(selected_stock, selected_sector, experiment_dir, selected_signal_file)
        df_plot = df_plot.reset_index(drop=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(
            x=df_plot.index, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'],
            name='K线', increasing_line_color='red', decreasing_line_color='green'
        ), row=1, col=1)
        if 'predicted_close' in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot['predicted_close'], mode='lines',
                name='预测收盘价', line=dict(color='cyan', width=2, dash='dot')
            ), row=1, col=1)
        df_plot['MA5'] = df_plot['close'].rolling(window=5).mean()
        df_plot['MA10'] = df_plot['close'].rolling(window=10).mean()
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA5'], mode='lines', name='MA5', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA10'], mode='lines', name='MA10', line=dict(color='purple', width=1)), row=1, col=1)
        buy_signals = df_plot[df_plot['predicted_action'] == 2]
        sell_signals = df_plot[df_plot['predicted_action'] == 0]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index, y=buy_signals['low'] * 0.98,
                mode='markers', marker=dict(symbol='triangle-up', color='red', size=10), name='预测买入'
            ), row=1, col=1)
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index, y=sell_signals['high'] * 1.02,
                mode='markers', marker=dict(symbol='triangle-down', color='green', size=10), name='预测卖出'
            ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=df_plot.index, y=df_plot['volume'], name='成交量',
            marker_color=np.where(df_plot['close'] >= df_plot['open'], 'red', 'green')
        ), row=2, col=1)
        num_points = len(df_plot)
        if num_points > 0:
            tick_indices = np.linspace(0, num_points - 1, 10, dtype=int)
            tick_values = df_plot.index[tick_indices].tolist()
            tick_texts = df_plot['date'].iloc[tick_indices].dt.strftime('%Y-%m-%d').tolist()
        else:
            tick_values = []
            tick_texts = []
        fig.update_layout(
            title_text=f'{selected_stock} ({selected_sector}) - {selected_signal_file.replace("_daily_signals.csv","")} 信号',
            xaxis_rangeslider_visible=False, height=800, template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                tickmode='array',
                tickvals=tick_values,
                ticktext=tick_texts,
                showgrid=True,
                rangeslider=dict(visible=False)
            ),
            xaxis2=dict(
                tickmode='array',
                tickvals=tick_values,
                ticktext=tick_texts,
                showgrid=True,
                rangeslider=dict(visible=False)
            )
        )
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)
        hover_texts = []
        num_points = len(df_plot)
        for i in range(num_points):
             row = df_plot.iloc[i]
             hover_text = f"日期: {row['date'].strftime('%Y-%m-%d')}<br>"
             hover_text += f"开盘: {row['open']:.2f}<br>"
             hover_text += f"最高: {row['high']:.2f}<br>"
             hover_text += f"最低: {row['low']:.2f}<br>"
             hover_text += f"收盘: {row['close']:.2f}<br>"
             if 'predicted_close' in row:
                 hover_text += f"预测收盘价: {row['predicted_close']:.2f}<br>"
             action = '持仓'
             if row.get('predicted_action') == 2: action = '买入'
             elif row.get('predicted_action') == 0: action = '卖出'
             hover_text += f"预测操作: {action}"
             hover_texts.append(hover_text)
        fig.data[0].hovertext = hover_texts
        fig.data[0].hoverinfo = 'text'
        return fig, ""
    except Exception as e:
        error_message = f"发生错误: {e}\n{traceback.format_exc()}"
        print(error_message)
        return go.Figure(), error_message

if __name__ == '__main__':
    print("启动仪表盘...")
    app.run(debug=True)