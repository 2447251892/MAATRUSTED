# 文件名: time_series_maa.py

from MAA_base import MAABase
import torch
import torch.nn as nn
import numpy as np
from functools import wraps
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from utils.multiGAN_trainer_disccls import train_multi_gan
from typing import List, Optional
import models
import os
import time
import glob
from utils.evaluate_visualization import evaluate_best_models
import joblib
import sys
import logging
import traceback
from tqdm import tqdm


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time();
        result = func(*args, **kwargs);
        end_time = time.time()
        print(f"MAA_time_series - 方法 '{func.__name__}' 执行耗时: {end_time - start_time:.4f} 秒")
        return result

    return wrapper


class CAE_for_pretrain(nn.Module):
    def __init__(self, input_height, input_width):
        super().__init__()
        self.encoder = models.Generator_mpd(input_height, input_width, num_classes=3).cae_encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.AdaptiveAvgPool2d((input_height, input_width)),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def generate_labels(y):
    y = np.array(y).flatten();
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
    def __init__(self, args, N_pairs, batch_size, num_epochs, generators_names, discriminators_names, ckpt_dir,
                 output_dir, window_sizes, initial_learning_rate=2e-5, do_distill_epochs=1, cross_finetune_epochs=5,
                 precise=torch.float32, device=None, seed=None, ckpt_path=None, gan_weights=None):
        super().__init__(N_pairs, batch_size, num_epochs, generators_names, discriminators_names, ckpt_dir, output_dir,
                         initial_learning_rate, precise, do_distill_epochs, cross_finetune_epochs, device, seed,
                         ckpt_path)
        self.args = args;
        self.window_sizes = window_sizes;
        self.generator_dict = {};
        self.discriminator_dict = {"default": models.Discriminator3}
        for name in dir(models):
            obj = getattr(models, name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                lname = name.lower()
                if "generator" in lname:
                    self.generator_dict[lname.replace("generator_", "")] = obj
                elif "discriminator" in lname and name.startswith("Discriminator"):
                    self.discriminator_dict[lname.replace("discriminator", "")] = obj
        self.gan_weights = gan_weights;
        self.init_hyperparameters()

    def pretrain_cae_if_needed(self, all_stock_files: List[str], cae_ckpt_path: str, pretrain_epochs: int):
        all_datasets = []
        window_size = self.window_sizes[0]

        temp_df = pd.read_csv(all_stock_files[0])
        temp_feature_cols = [col for col in temp_df.columns if col not in ['date', 'direction']]
        num_features = len(temp_feature_cols)

        for stock_file in tqdm(all_stock_files, desc="为CAE预训练加载数据"):
            df = pd.read_csv(stock_file)
            features = df[temp_feature_cols].values

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = scaler.fit_transform(features)

            images = []
            for i in range(window_size, len(scaled_features)):
                image_data = scaled_features[i - window_size: i, :].reshape(1, window_size, num_features)
                images.append(image_data)

            if images:
                all_datasets.append(TensorDataset(torch.from_numpy(np.array(images)).float()))

        if not all_datasets:
            print("错误: 未能为CAE预训练创建任何数据。")
            return

        full_dataset = ConcatDataset(all_datasets)
        dataloader = DataLoader(full_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        cae_model = CAE_for_pretrain(input_height=window_size, input_width=num_features).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(cae_model.parameters(), lr=1e-3)

        cae_model.train()

        with tqdm(total=pretrain_epochs * len(dataloader), desc="CAE预训练") as pbar:
            for epoch in range(pretrain_epochs):
                epoch_loss = 0.0
                for batch_idx, batch in enumerate(dataloader):
                    images = batch[0].to(self.device)

                    optimizer.zero_grad()
                    outputs = cae_model(images)
                    loss = criterion(outputs, images)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    pbar.set_postfix(
                        Epoch=f'{epoch + 1}/{pretrain_epochs}',
                        Loss=f'{epoch_loss / (batch_idx + 1):.6f}'
                    )
                    pbar.update(1)

        os.makedirs(os.path.dirname(cae_ckpt_path), exist_ok=True)
        torch.save(cae_model.encoder.state_dict(), cae_ckpt_path)

    @log_execution_time
    def process_data(self, train_csv_path, predict_csv_path, target_column, exclude_columns):
        print(f"正在加载预处理好的训练数据: {train_csv_path}");
        train_df = pd.read_csv(train_csv_path);
        self.predict_df = pd.read_csv(predict_csv_path)
        print(f"成功加载训练数据 ({len(train_df)}行) 和预测数据 ({len(self.predict_df)}行)。")
        feature_columns = [col for col in train_df.columns if col != target_column and col not in exclude_columns]
        print(f"目标列: {target_column}\n已自动识别 {len(feature_columns)} 个特征列。");
        self.feature_columns = feature_columns
        x_from_train_df, y_from_train_df = train_df[feature_columns].values, train_df[[target_column]].values
        train_ratio, val_ratio, _ = self.args.train_val_test_split;
        total_len = len(train_df)
        train_end_idx, val_end_idx = int(total_len * train_ratio), int(total_len * (train_ratio + val_ratio))
        train_x_raw, val_x_raw, test_x_raw = x_from_train_df[:train_end_idx], x_from_train_df[
                                                                              train_end_idx:val_end_idx], x_from_train_df[
                                                                                                          val_end_idx:]
        train_y_raw, val_y_raw, test_y_raw = y_from_train_df[:train_end_idx], y_from_train_df[
                                                                              train_end_idx:val_end_idx], y_from_train_df[
                                                                                                          val_end_idx:]
        self.train_size = len(train_x_raw);
        print(
            f"数据按 {train_ratio}:{val_ratio}:{1 - train_ratio - val_ratio:.1f} 比例划分: 训练集={len(train_x_raw)}, 验证集={len(val_x_raw)}, 测试集={len(test_x_raw)}")
        self.x_scalers, self.y_scaler = [MinMaxScaler(feature_range=(0, 1)) for _ in range(self.N)], MinMaxScaler(
            feature_range=(0, 1))
        self.train_x_list = [s.fit_transform(train_x_raw) for s in self.x_scalers];
        self.train_y = self.y_scaler.fit_transform(train_y_raw)
        self.val_x_list = [s.transform(val_x_raw) for s in self.x_scalers];
        self.val_y = self.y_scaler.transform(val_y_raw)
        self.test_x_list = [s.transform(test_x_raw) for s in self.x_scalers];
        self.test_y = self.y_scaler.transform(test_y_raw)
        self.train_labels, self.val_labels, self.test_labels = generate_labels(self.train_y), generate_labels(
            self.val_y), generate_labels(self.test_y)
        print(
            f"数据加载、划分和归一化完成。训练集: {len(self.train_y)} 条, 验证集: {len(self.val_y)} 条, 测试集: {len(self.test_y)} 条。")

    def create_sequences_combine(self, x, y, label, window_size, start):
        x_seq, y_reg, y_gan, label_gan = [], [], [], []
        x_img = []
        num_features = x.shape[1]
        for i in range(start, x.shape[0]):
            x_seq.append(x[i - window_size: i, :])
            y_reg.append(y[i])
            y_gan.append(y[i - window_size: i + 1])
            label_gan.append(label[i - window_size: i + 1])
            image_data = x[i - window_size: i, :].reshape(1, window_size, num_features)
            x_img.append(image_data)
        return (
            torch.from_numpy(np.array(x_seq)).float(),
            torch.from_numpy(np.array(x_img)).float(),
            torch.from_numpy(np.array(y_reg)).float(),
            torch.from_numpy(np.array(y_gan)).float(),
            torch.from_numpy(np.array(label_gan)).long()
        )

    @log_execution_time
    def init_dataloader(self):
        ws = self.window_sizes
        train_data = [self.create_sequences_combine(self.train_x_list[i], self.train_y, self.train_labels, w, ws[-1])
                      for i, w in enumerate(ws)]
        val_data = [self.create_sequences_combine(self.val_x_list[i], self.val_y, self.val_labels, w, ws[-1]) for i, w
                    in enumerate(ws)]
        test_data = [self.create_sequences_combine(self.test_x_list[i], self.test_y, self.test_labels, w, ws[-1]) for
                     i, w in enumerate(ws)]
        self.train_x_all = [x_seq.to(self.device) for x_seq, _, _, _, _ in train_data]
        self.train_x_img_all = [x_img.to(self.device) for _, x_img, _, _, _ in train_data]
        self.train_y_all = train_data[0][2]
        self.train_y_gan_all = [y_gan.to(self.device) for _, _, _, y_gan, _ in train_data]
        self.train_label_gan_all = [l_gan.to(self.device) for _, _, _, _, l_gan in train_data]
        self.val_x_all = [x_seq.to(self.device) for x_seq, _, _, _, _ in val_data]
        self.val_x_img_all = [x_img.to(self.device) for _, x_img, _, _, _ in val_data]
        self.val_y_all = val_data[0][2]
        self.val_y_gan_all = [y_gan.to(self.device) for _, _, _, y_gan, _ in val_data]
        self.val_label_gan_all = [l_gan.to(self.device) for _, _, _, _, l_gan in val_data]
        self.test_x_all = [x_seq.to(self.device) for x_seq, _, _, _, _ in test_data]
        self.test_x_img_all = [x_img.to(self.device) for _, x_img, _, _, _ in test_data]
        self.test_y_all = test_data[0][2]
        self.test_y_gan_all = [y_gan.to(self.device) for _, _, _, y_gan, _ in test_data]
        self.test_label_gan_all = [l_gan.to(self.device) for _, _, _, _, l_gan in test_data]
        self.dataloaders = []
        for i in range(self.N):
            is_mpd = "mpd" in self.generator_names[i]
            x_data_for_loader = self.train_x_img_all[i] if is_mpd else self.train_x_all[i]
            y_gan_data = self.train_y_gan_all[i]
            l_gan_data = self.train_label_gan_all[i]
            self.dataloaders.append(
                DataLoader(TensorDataset(x_data_for_loader, y_gan_data, l_gan_data),
                           batch_size=self.batch_size,
                           shuffle=("transformer" in self.generator_names[i] or "mpd" in self.generator_names[i]),
                           generator=torch.manual_seed(self.seed),
                           drop_last=True)
            )

    @log_execution_time
    def init_model(self, num_cls):
        self.generators, self.discriminators = [], [];
        if len(self.generator_names) != self.N or len(self.window_sizes) != self.N: sys.exit(1)
        for i, name in enumerate(self.generator_names):
            if not hasattr(self, 'train_x_all') or not self.train_x_all or i >= len(self.train_x_all): sys.exit(1)

            input_dim, y_dim = self.train_x_all[i].shape[-1], self.train_y_all.shape[-1]
            GenClass = self.generator_dict.get(name);
            if GenClass is None: sys.exit(1)

            generator_instance = None
            if name == "mpd":
                window_size = self.window_sizes[i]
                num_features = len(self.feature_columns)
                generator_instance = GenClass(input_height=window_size, input_width=num_features, num_classes=num_cls)

                # ==================== 修正点: 使用个股专属的路径 ====================
                # args.cae_ckpt_path 现在是由 run_multi_gan 传入的、包含完整路径的参数
                cae_ckpt_path_stock = self.args.cae_ckpt_path
                # ========================== 修正结束 ==========================

                if os.path.exists(cae_ckpt_path_stock):
                    try:
                        generator_instance.cae_encoder.load_state_dict(
                            torch.load(cae_ckpt_path_stock, map_location=self.device))
                        print(f"成功为 Generator_mpd (G{i + 1}) 加载了预训练的CAE编码器权重。")
                    except Exception as e:
                        print(f"警告: 为 G{i + 1} 加载预训练CAE编码器权重失败: {e}")
                else:
                    if self.args.pretrain_cae:
                        print(
                            f"信息: 未找到该股票的预训练权重 '{cae_ckpt_path_stock}'，将在微调时从零开始训练CAE编码器部分。")

            elif name == "transformer":
                generator_instance = GenClass(input_dim=input_dim, output_len=y_dim)
            elif name == "transformer_deep":
                generator_instance = GenClass(input_dim=input_dim, output_len=y_dim)
            elif name == "dct":
                generator_instance = GenClass(input_dim=input_dim, out_size=y_dim, num_classes=num_cls)
            elif name == "rnn":
                generator_instance = GenClass(input_size=input_dim)
            else:
                generator_instance = GenClass(input_size=input_dim, out_size=y_dim)

            if generator_instance:
                self.generators.append(generator_instance.to(self.device))

            DisClass = self.discriminator_dict.get("default");
            if DisClass is None: sys.exit(1)
            self.discriminators.append(
                DisClass(input_dim=self.window_sizes[i] + 1, out_size=y_dim, num_cls=num_cls).to(self.device))

    def init_hyperparameters(self):
        self.init_GDweight = [[1.0 if i == j else 0.0 for j in range(self.N)] + [1.0] for i in range(self.N)]
        self.final_GDweight = self.gan_weights if self.gan_weights else [[round(1.0 / self.N, 3)] * self.N + [1.0] for _
                                                                         in range(self.N)]

    def train(self, logger, date_series=None):
        is_mpd_run = "mpd" in self.generator_names[0]
        train_x_to_pass = self.train_x_img_all if is_mpd_run else self.train_x_all
        val_x_to_pass = self.val_x_img_all if is_mpd_run else self.val_x_all

        return train_multi_gan(
            args=self.args,
            generators=self.generators,
            discriminators=self.discriminators,
            dataloaders=self.dataloaders,
            window_sizes=self.window_sizes,
            y_scaler=self.y_scaler,
            train_xes=train_x_to_pass,
            train_y=self.train_y_all,
            val_xes=val_x_to_pass,
            val_y=self.val_y_all,
            val_y_gan=self.val_y_gan_all,
            val_label_gan=self.val_label_gan_all,
            output_dir=self.output_dir,
            device=self.device,
            init_GDweight=self.init_GDweight,
            final_GDweight=self.final_GDweight,
            logger=logger,
            date_series=date_series
        )

    def save_models(self, best_model_state):
        gen_dir = os.path.join(self.ckpt_dir, "generators");
        disc_dir = os.path.join(self.ckpt_dir, "discriminators");
        os.makedirs(gen_dir, exist_ok=True);
        os.makedirs(disc_dir, exist_ok=True)
        for i, gen_name in enumerate(self.generator_names):
            if i < len(best_model_state) and best_model_state[i]: torch.save(best_model_state[i], os.path.join(gen_dir,
                                                                                                               f"{i + 1}_{gen_name}.pt"))
        for i, disc in enumerate(self.discriminators):
            if i < len(self.discriminators): torch.save(disc.state_dict(),
                                                        os.path.join(disc_dir, f"{i + 1}_Discriminator3.pt"))
        print(f"所有模型已成功保存至: {self.ckpt_dir}")

    def save_predictions_to_csv(self, date_series=None):
        print("\n--- 开始生成并保存真实值 vs. 预测值对比CSV文件 (基于测试集) ---")
        if not hasattr(self, 'generators') or not hasattr(self, 'test_x_all') or not hasattr(self,
                                                                                             'test_y_all') or not self.test_x_all: return

        is_mpd_run = "mpd" in self.generator_names[0]
        test_x_to_use = self.test_x_img_all if is_mpd_run else self.test_x_all

        with torch.no_grad():
            for i, gen in enumerate(self.generators):
                gen.eval();
                if i >= len(test_x_to_use): continue
                try:
                    y_pred_gen, _ = gen(test_x_to_use[i]);
                    y_pred_norm = y_pred_gen.cpu().numpy().reshape(-1, 1)
                    true_y_segment = self.test_y_all[-len(y_pred_gen):];
                    y_true_norm = true_y_segment.cpu().numpy().reshape(-1, 1)
                    y_pred = self.y_scaler.inverse_transform(y_pred_norm).flatten();
                    y_true = self.y_scaler.inverse_transform(y_true_norm).flatten()
                    df_out = pd.DataFrame({'true': y_true, 'pred': y_pred})
                    if date_series is not None:
                        max_ws = max(self.window_sizes) if self.window_sizes else 0;
                        date_aligned = date_series.iloc[max_ws:].reset_index(drop=True)
                        ts_seq_len, vs_seq_len = len(self.train_y_all), len(self.val_y_all)
                        if len(date_aligned) >= (ts_seq_len + vs_seq_len + len(y_pred)):
                            test_dates = date_aligned.iloc[ts_seq_len + vs_seq_len:];
                            dates_csv = test_dates.iloc[-len(y_pred):].reset_index(drop=True)
                            df_out['date'] = dates_csv;
                            df_out = df_out[['date', 'true', 'pred']]
                        else:
                            logging.warning(f"G{i + 1}: 日期序列长度不足，无法分配日期。")
                    csv_dir = os.path.join(self.output_dir, "true2pred_csv");
                    os.makedirs(csv_dir, exist_ok=True)
                    out_path = os.path.join(csv_dir, f'predictions_gen_{i + 1}_{self.generator_names[i]}.csv');
                    df_out.to_csv(out_path, index=False)
                    print(f"已保存 G{i + 1} 的真实值与预测值对比: {out_path}")
                except Exception as e:
                    print(f"错误: 为 G{i + 1} 生成预测CSV时出错: {e}");
                    traceback.print_exc()

    def pred(self, date_series=None):
        print(f"开始使用以下路径的模型进行预测: {self.ckpt_dir}");
        gen_dir = os.path.join(self.ckpt_dir, "generators");
        if not os.path.isdir(gen_dir): raise FileNotFoundError(f"找不到 'generators' 文件夹: {gen_dir}")
        best_model_state = [None] * self.N;
        loaded_count = 0
        for i, gen_name in enumerate(self.generator_names):
            save_path = os.path.join(gen_dir, f"{i + 1}_{gen_name}.pt")
            if os.path.exists(save_path):
                try:
                    state_dict = torch.load(save_path, map_location=self.device)
                    expected_class_name = self.generators[i].__class__.__name__.lower()
                    if i < len(self.generators) and expected_class_name.replace("generator_", "") == gen_name.lower():
                        self.generators[i].load_state_dict(state_dict);
                        best_model_state[i] = state_dict;
                        loaded_count += 1
                except Exception as e:
                    print(f"错误: 加载检查点 {save_path} 失败: {e}");
                    traceback.print_exc()
        if loaded_count == 0: print("错误: 未能加载任何模型，无法进行预测。"); return None
        if not all(hasattr(self, attr) for attr in
                   ['train_x_all', 'test_x_all', 'train_y_all', 'test_y_all', 'y_scaler']): return None

        is_mpd_run = "mpd" in self.generator_names[0]
        train_xes_to_eval = self.train_x_img_all if is_mpd_run else self.train_x_all
        test_xes_to_eval = self.test_x_img_all if is_mpd_run else self.test_x_all

        results = evaluate_best_models(self.generators, best_model_state,
                                       train_xes_to_eval, self.train_y_all,
                                       test_xes_to_eval, self.test_y_all,
                                       self.y_scaler, self.output_dir,
                                       self.window_sizes, date_series=date_series)
        self.save_predictions_to_csv(date_series=date_series);
        return results

    def save_scalers(self):
        if not hasattr(self, 'x_scalers') or not self.x_scalers or not hasattr(self, 'y_scaler'): return
        try:
            joblib.dump(self.x_scalers[0], os.path.join(self.output_dir, 'x_scaler.gz'));
            joblib.dump(self.y_scaler, os.path.join(self.output_dir, 'y_scaler.gz'))
            print(f"Scaler 已成功保存至: {self.output_dir}")
        except Exception as e:
            print(f"错误: 保存 scaler 失败: {e}");
            traceback.print_exc()

    def generate_and_save_daily_signals(self, best_model_state, predict_csv_path):
        if not all(hasattr(self, attr) for attr in
                   ['x_scalers', 'y_scaler', 'feature_columns', 'generators', 'train_size', 'val_y',
                    'window_sizes']): return
        if not best_model_state or not any(s is not None for s in best_model_state): return
        print("\n--- 开始为所有模型生成每日预测信号 ---");
        df_predict = pd.read_csv(predict_csv_path)
        x_scaler, y_scaler = self.x_scalers[0], self.y_scaler
        val_size = len(self.val_y);
        loop_start_index = self.train_size + val_size + self.window_sizes[-1]
        if loop_start_index >= len(df_predict): print(f"警告: 第一个测试预测索引 {loop_start_index} 超出范围。"); return
        for i, state in enumerate(best_model_state):
            if state is None: continue
            gen_name, window_size, generator = self.generator_names[i], self.window_sizes[i], self.generators[i]
            is_mpd_model = "mpd" in gen_name
            try:
                generator.load_state_dict(state);
                generator.eval()
            except Exception as e:
                print(f"错误: 加载 G{i + 1} ({gen_name}) 状态失败: {e}");
                continue
            print(f"正在处理模型: G{i + 1} ({gen_name})，窗口大小: {window_size}");
            signals = []
            for j in range(loop_start_index, len(df_predict)):
                if j < max(window_size, loop_start_index): continue
                df_segment = df_predict.iloc[j - window_size: j];
                if len(df_segment) < window_size: continue
                sequence_data = df_segment[self.feature_columns].values
                if np.isnan(sequence_data).any(): continue
                try:
                    scaled_sequence = x_scaler.transform(sequence_data)
                    if is_mpd_model:
                        num_features = scaled_sequence.shape[1]
                        input_data = scaled_sequence.reshape(1, 1, window_size, num_features)
                    else:
                        input_data = np.array([scaled_sequence])

                    input_tensor = torch.from_numpy(input_data).float().to(self.device)

                    with torch.no_grad():
                        gen_output, logits = generator(input_tensor);
                        predicted_action = logits.argmax(dim=1).item()
                        predicted_close_real = y_scaler.inverse_transform(gen_output.cpu().numpy()).flatten()[0]
                    signals.append({'date': df_predict.iloc[j]['date'], 'predicted_action': predicted_action,
                                    'predicted_close': predicted_close_real})
                except Exception as e:
                    print(f"错误: 在日期索引 {j} 生成 G{i + 1} ({gen_name}) 信号时出错: {e}");
            if signals:
                df_signals = pd.DataFrame(signals);
                signal_filepath = os.path.join(self.output_dir, f'G{i + 1}_{gen_name}_daily_signals.csv')
                df_signals.to_csv(signal_filepath, index=False, float_format='%.4f');
                print(f"已保存每日信号文件: {signal_filepath}")
            else:
                print(f"警告: G{i + 1} ({gen_name}) 没有生成任何有效信号。")

    def distill(self):
        pass

    def visualize_and_evaluate(self):
        pass

    def init_history(self):
        pass