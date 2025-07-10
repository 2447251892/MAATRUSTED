# 文件名: time_series_maa.py

from MAA_base import MAABase
import torch
import numpy as np
from functools import wraps
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from utils.multiGAN_trainer_disccls import train_multi_gan
from typing import List, Optional
import models
import os
import time
import glob
from utils.evaluate_visualization import evaluate_best_models
import joblib


def log_execution_time(func):
    """装饰器：记录函数的执行时间。"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        func_name = func.__name__
        print(f"MAA_time_series - 方法 '{func_name}' 执行耗时: {elapsed_time:.4f} 秒")
        return result

    return wrapper


def generate_labels(y):
    """根据涨跌生成三分类标签。"""
    y = np.array(y).flatten()
    labels = [1]
    for i in range(1, len(y)):
        if y[i] > y[i - 1]:
            labels.append(2)
        elif y[i] < y[i - 1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)


class MAA_time_series(MAABase):
    def __init__(self, args, N_pairs: int, batch_size: int, num_epochs: int,
                 generators_names: List, discriminators_names: Optional[List],
                 ckpt_dir: str, output_dir: str,
                 window_sizes: int,
                 initial_learning_rate: float = 2e-5,
                 train_split: float = 0.8,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 precise=torch.float32,
                 device=None,
                 seed: int = None,
                 ckpt_path: str = None,
                 gan_weights=None,
                 ):
        super().__init__(N_pairs, batch_size, num_epochs,
                         generators_names, discriminators_names,
                         ckpt_dir, output_dir,
                         initial_learning_rate,
                         train_split,
                         precise,
                         do_distill_epochs, cross_finetune_epochs,
                         device,
                         seed,
                         ckpt_path)

        self.args = args
        self.window_sizes = window_sizes
        self.generator_dict = {}
        self.discriminator_dict = {"default": models.Discriminator3}

        for name in dir(models):
            obj = getattr(models, name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                lname = name.lower()
                if "generator" in lname:
                    key = lname.replace("generator_", "")
                    self.generator_dict[key] = obj
                elif "discriminator" in lname:
                    key = lname.replace("discriminator", "")
                    self.discriminator_dict[key] = obj

        self.gan_weights = gan_weights
        self.init_hyperparameters()

    @log_execution_time
    def process_data(self, train_csv_path: str, predict_csv_path: str, target_column: str, exclude_columns: list):
        """加载已预处理的数据。"""
        print(f"正在加载预处理好的训练数据: {train_csv_path}")
        try:
            train_df = pd.read_csv(train_csv_path)
            self.predict_df = pd.read_csv(predict_csv_path)
            print(f"成功加载训练数据 ({len(train_df)}行) 和预测数据 ({len(self.predict_df)}行)。")
        except FileNotFoundError as e:
            print(f"错误: 找不到数据文件 {e.filename}。请确保 get_stock_data.py 已成功运行。")
            raise

        all_columns = train_df.columns.tolist()
        feature_columns = [col for col in all_columns if col != target_column and col not in exclude_columns]

        print(f"目标列: {target_column}")
        print(f"已自动识别 {len(feature_columns)} 个特征列。")
        self.feature_columns = feature_columns  # 保存特征列列表以备后用

        x_from_train_df = train_df[feature_columns].values
        y_from_train_df = train_df[[target_column]].values

        train_size = int(len(train_df) * self.train_split)
        train_x_raw, test_x_raw = x_from_train_df[:train_size], x_from_train_df[train_size:]
        train_y_raw, test_y_raw = y_from_train_df[:train_size], y_from_train_df[train_size:]

        self.x_scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(self.N)]
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))

        self.train_x_list = [scaler.fit_transform(train_x_raw) for scaler in self.x_scalers]
        self.test_x_list = [scaler.transform(test_x_raw) for scaler in self.x_scalers]
        self.train_y = self.y_scaler.fit_transform(train_y_raw)
        self.test_y = self.y_scaler.transform(test_y_raw)

        self.train_labels = generate_labels(self.train_y)
        self.test_labels = generate_labels(self.test_y)

        print(f"数据加载、划分和归一化完成。训练集: {len(self.train_y)} 条, 测试集: {len(self.test_y)} 条。")

    def create_sequences_combine(self, x, y, label, window_size, start):
        x_, y_, y_gan, label_gan = [], [], [], []
        for i in range(start, x.shape[0]):
            x_.append(x[i - window_size: i, :])
            y_.append(y[i])
            y_gan.append(y[i - window_size: i + 1])
            label_gan.append(label[i - window_size: i + 1])
        return (torch.from_numpy(np.array(x_)).float(),
                torch.from_numpy(np.array(y_)).float(),
                torch.from_numpy(np.array(y_gan)).float(),
                torch.from_numpy(np.array(label_gan)).float())

    @log_execution_time
    def init_dataloader(self):
        train_data_list = [self.create_sequences_combine(self.train_x_list[i], self.train_y, self.train_labels, w,
                                                         self.window_sizes[-1]) for i, w in
                           enumerate(self.window_sizes)]
        test_data_list = [
            self.create_sequences_combine(self.test_x_list[i], self.test_y, self.test_labels, w, self.window_sizes[-1])
            for i, w in enumerate(self.window_sizes)]

        self.train_x_all = [x.to(self.device) for x, _, _, _ in train_data_list]
        self.train_y_all = train_data_list[0][1]
        self.train_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in train_data_list]
        self.train_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in train_data_list]

        self.test_x_all = [x.to(self.device) for x, _, _, _ in test_data_list]
        self.test_y_all = test_data_list[0][1]
        self.test_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in test_data_list]
        self.test_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in test_data_list]

        self.dataloaders = [DataLoader(TensorDataset(x, y_gan, label_gan), batch_size=self.batch_size,
                                       shuffle=("transformer" in self.generator_names[i]),
                                       generator=torch.manual_seed(self.seed), drop_last=True) for
                            i, (x, y_gan, label_gan) in
                            enumerate(zip(self.train_x_all, self.train_y_gan_all, self.train_label_gan_all))]

    def init_model(self, num_cls):
        self.generators, self.discriminators = [], []
        for i, name in enumerate(self.generator_names):
            x, y_dim = self.train_x_all[i], self.train_y_all.shape[-1]
            GenClass = self.generator_dict[name]
            self.generators.append(
                GenClass(x.shape[-1], output_len=y_dim).to(self.device) if "transformer" in name else GenClass(
                    x.shape[-1], y_dim).to(self.device))
            DisClass = self.discriminator_dict["default"]
            self.discriminators.append(
                DisClass(self.window_sizes[i] + 1, out_size=y_dim, num_cls=num_cls).to(self.device))

    def init_hyperparameters(self):
        self.init_GDweight = [[1.0 if i == j else 0.0 for j in range(self.N)] + [1.0] for i in range(self.N)]
        self.final_GDweight = self.gan_weights if self.gan_weights else [[round(1.0 / self.N, 3)] * self.N + [1.0] for _
                                                                         in range(self.N)]

    def train(self, logger, date_series=None):
        return train_multi_gan(self.args, self.generators, self.discriminators, self.dataloaders,
                               self.window_sizes, self.y_scaler, self.train_x_all, self.train_y_all,
                               self.test_x_all, self.test_y_all, self.test_label_gan_all,
                               self.do_distill_epochs, self.cross_finetune_epochs, self.num_epochs,
                               self.output_dir, self.device, init_GDweight=self.init_GDweight,
                               final_GDweight=self.final_GDweight, logger=logger,
                               date_series=date_series)

    def save_models(self, best_model_state):
        gen_dir = os.path.join(self.ckpt_dir, "generators")
        disc_dir = os.path.join(self.ckpt_dir, "discriminators")
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(disc_dir, exist_ok=True)
        for i, gen_name in enumerate(self.generator_names):
            if best_model_state[i]:
                torch.save(best_model_state[i], os.path.join(gen_dir, f"{i + 1}_{gen_name}.pt"))
        for i, disc in enumerate(self.discriminators):
            torch.save(disc.state_dict(), os.path.join(disc_dir, f"{i + 1}_Discriminator3.pt"))
        print(f"所有模型已成功保存至: {self.ckpt_dir}")

    def save_predictions_to_csv(self, date_series=None):
        """将每个生成器的真实值与预测值保存到CSV文件。"""
        print("\n--- 开始生成并保存真实值 vs. 预测值对比CSV文件 ---")
        if not hasattr(self, 'generators') or not hasattr(self, 'test_x_all') or not hasattr(self, 'test_y_all'):
            print("警告: 缺少必要的评估数据，无法生成CSV。")
            return
        with torch.no_grad():
            for i, gen in enumerate(self.generators):
                gen.eval()
                if i >= len(self.test_x_all): continue
                try:
                    y_pred_gen, _ = gen(self.test_x_all[i])
                    y_pred_norm = y_pred_gen.cpu().numpy().reshape(-1, 1)
                    y_true_norm = self.test_y_all.cpu().numpy().reshape(-1, 1)
                    y_pred = self.y_scaler.inverse_transform(y_pred_norm).flatten()
                    y_true = self.y_scaler.inverse_transform(y_true_norm).flatten()

                    y_true_matched = y_true[-len(y_pred):]
                    df_out = pd.DataFrame({'true': y_true_matched, 'pred': y_pred})

                    if date_series is not None:
                        total_size_in_series = len(date_series)
                        test_size_processed = len(self.test_y_all)
                        test_start_idx_in_series = total_size_in_series - test_size_processed

                        if test_start_idx_in_series >= 0:
                            test_dates = date_series.iloc[test_start_idx_in_series:].reset_index(drop=True)
                            dates_for_csv = test_dates.iloc[-len(y_pred):].reset_index(drop=True)
                            df_out['date'] = dates_for_csv
                            df_out = df_out[['date', 'true', 'pred']]
                        else:
                            print(f"警告: 无法为生成器 {i + 1} 的CSV文件分配日期。")

                    csv_save_dir = os.path.join(self.output_dir, "true2pred_csv")
                    os.makedirs(csv_save_dir, exist_ok=True)
                    out_path = os.path.join(csv_save_dir, f'predictions_gen_{i + 1}_{self.generator_names[i]}.csv')
                    df_out.to_csv(out_path, index=False)
                    print(f"已保存生成器 {i + 1} 的真实值与预测值对比: {out_path}")
                except Exception as e:
                    print(f"错误: 为生成器 {i + 1} 生成预测CSV时出错: {e}")
                    import traceback
                    traceback.print_exc()

    def pred(self, date_series=None):
        """执行预测流程。"""
        current_model_path = self.ckpt_dir if self.ckpt_path == "auto" else self.ckpt_path
        print(f"开始使用以下路径的模型进行预测: {current_model_path}")
        gen_dir = os.path.join(current_model_path, "generators")
        if not os.path.isdir(gen_dir): raise FileNotFoundError(f"找不到 'generators' 文件夹: {gen_dir}")
        best_model_state = [None] * self.N
        for i, gen_name in enumerate(self.generator_names):
            save_path = os.path.join(gen_dir, f"{i + 1}_{gen_name}.pt")
            if os.path.exists(save_path):
                state_dict = torch.load(save_path, map_location=self.device)
                self.generators[i].load_state_dict(state_dict)
                best_model_state[i] = state_dict
        if not any(best_model_state):
            print("错误: 未能加载任何模型，无法进行预测。")
            return None
        results = evaluate_best_models(self.generators, best_model_state, self.train_x_all, self.train_y_all,
                                       self.test_x_all, self.test_y_all, self.y_scaler,
                                       self.output_dir, date_series=date_series)
        self.save_predictions_to_csv(date_series=date_series)
        return results

    def save_scalers(self):
        """保存数据归一化 scaler。"""
        if not hasattr(self, 'x_scalers') or not hasattr(self, 'y_scaler'):
            print("错误: Scalers 尚未初始化，无法保存。")
            return

        try:
            x_scaler_path = os.path.join(self.output_dir, 'x_scaler.gz')
            joblib.dump(self.x_scalers[0], x_scaler_path)

            y_scaler_path = os.path.join(self.output_dir, 'y_scaler.gz')
            joblib.dump(self.y_scaler, y_scaler_path)
            print(f"Scaler 已成功保存至: {self.output_dir}")
        except Exception as e:
            print(f"错误: 保存 scaler 失败: {e}")

    def generate_and_save_daily_signals(self, best_model_state, predict_csv_path):
        """
        加载最佳模型，对完整的预测数据进行滚动预测，并保存每日信号。
        """
        if not hasattr(self, 'x_scalers'):
            print("错误: 缺少 x_scalers，无法进行信号生成。请先运行 process_data。")
            return

        if not best_model_state or not any(s is not None for s in best_model_state):
            print("没有可用的最佳模型状态，跳过每日信号生成。")
            return

        print("\n--- 开始为所有模型生成每日预测信号 ---")

        try:
            df_predict = pd.read_csv(predict_csv_path)
        except FileNotFoundError:
            print(f"错误：找不到预测数据文件 {predict_csv_path}。")
            return

        x_scaler = self.x_scalers[0]

        for i, state in enumerate(best_model_state):
            if state is None:
                continue

            gen_name = self.generator_names[i]
            window_size = self.window_sizes[i]
            generator = self.generators[i]
            generator.load_state_dict(state)
            generator.eval()

            print(f"正在处理模型: G{i + 1} ({gen_name})，窗口大小: {window_size}")

            signals = []

            # 使用在 process_data 中保存的特征列
            feature_columns = self.feature_columns

            for j in range(window_size, len(df_predict)):
                df_segment = df_predict.iloc[j - window_size: j]

                sequence_data = df_segment[feature_columns].values
                scaled_sequence = x_scaler.transform(sequence_data)
                input_tensor = torch.from_numpy(np.array([scaled_sequence])).float().to(self.device)

                with torch.no_grad():
                    _, logits = generator(input_tensor)
                    prediction = logits.argmax(dim=1).item()

                signal_date = df_predict.iloc[j]['date']
                signals.append({'date': signal_date, 'predicted_action': prediction})

            if signals:
                df_signals = pd.DataFrame(signals)
                signal_filename = f'G{i + 1}_{gen_name}_daily_signals.csv'
                signal_filepath = os.path.join(self.output_dir, signal_filename)
                df_signals.to_csv(signal_filepath, index=False)
                print(f"已保存每日信号文件: {signal_filepath}")

    def distill(self):
        pass

    def visualize_and_evaluate(self):
        pass

    def init_history(self):
        pass