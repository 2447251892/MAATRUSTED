# 文件名: models/model_with_clsdisc.py

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, model_dim)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)


class Generator_mpd(nn.Module):
    def __init__(self, input_height, input_width, num_classes,
                 cae_out_channels=[16, 32, 64], cae_kernel_size=3,
                 feature_size=512, num_layers=2, num_heads=16,
                 dropout=0.1, output_len=1):
        super().__init__()
        self.cae_encoder = nn.Sequential()
        in_channels = 1
        for i, out_ch in enumerate(cae_out_channels):
            self.cae_encoder.add_module(f"conv_{i + 1}",
                                        nn.Conv2d(in_channels, out_ch, kernel_size=cae_kernel_size, padding='same'))
            self.cae_encoder.add_module(f"relu_{i + 1}", nn.ReLU(True))
            target_pool_size = (max(1, input_height // (2 ** (i + 1))), max(1, input_width // (2 ** (i + 1))))
            self.cae_encoder.add_module(f"pool_{i + 1}", nn.AdaptiveMaxPool2d(target_pool_size))
            in_channels = out_ch
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_height, input_width)
            cae_output_shape = self.cae_encoder(dummy_input).shape
            self.flattened_dim = cae_output_shape[1] * cae_output_shape[2] * cae_output_shape[3]
        self.to_transformer_input = nn.Linear(self.flattened_dim, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regression_head = nn.Linear(feature_size, output_len)
        self.classification_head = nn.Linear(feature_size, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, src):
        cae_features = self.cae_encoder(src)
        flattened_features = cae_features.view(-1, self.flattened_dim)
        transformer_input = self.to_transformer_input(flattened_features)
        transformer_input = transformer_input.unsqueeze(1)
        transformer_input = self.pos_encoder(transformer_input)
        transformer_output = self.transformer_encoder(transformer_input)
        last_feature = transformer_output[:, -1, :]
        gen_output = self.regression_head(last_feature)
        cls_output = self.classification_head(last_feature)
        return gen_output, cls_output


class Generator_dct(nn.Module):
    def __init__(self, input_dim, out_size, num_classes,
                 inception_channels=[96, 256, 384],
                 d_model=768, mlp_size=3072,
                 transformer_layers=12, transformer_heads=8,
                 transformer_dropout=0.3, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.out_size = out_size
        self.num_classes = num_classes
        self.activation = activation()
        self.d_model = d_model
        self.inception1 = nn.Linear(input_dim, inception_channels[0])
        self.inception2 = nn.Linear(inception_channels[0], inception_channels[1])
        self.inception3 = nn.Linear(inception_channels[1], inception_channels[2])
        self.sep_fc1 = nn.Linear(inception_channels[2], d_model)
        self.sep_fc2 = nn.Linear(d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=transformer_heads,
                                                   dim_feedforward=mlp_size,
                                                   dropout=transformer_dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.regression_head = nn.Linear(d_model, out_size)
        self.classification_head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, src):
        src = self.activation(self.inception1(src))
        src = self.activation(self.inception2(src))
        src = self.activation(self.inception3(src))
        src = self.activation(self.sep_fc1(src))
        src = self.activation(self.sep_fc2(src))
        src = self.pos_encoder(src)
        transformer_output = self.transformer_encoder(src)
        last_feature = transformer_output[:, -1, :]
        gen = self.regression_head(last_feature)
        cls = self.classification_head(last_feature)
        return gen, cls


class Generator_gru(nn.Module):
    def __init__(self, input_size, out_size, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear_2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.linear_3 = nn.Linear(hidden_dim // 4, out_size)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self, x):
        device = x.device
        h0 = torch.zeros(1, x.size(0), self.hidden_dim, device=device)
        out, _ = self.gru(x, h0)
        last_feature = self.dropout(out[:, -1, :])
        gen = self.linear_1(last_feature)
        gen = self.linear_2(gen)
        gen = self.linear_3(gen)
        cls = self.classifier(last_feature)
        return gen, cls


class Generator_lstm(nn.Module):
    def __init__(self, input_size, out_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.depth_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size,
                                    kernel_size=3, padding='same', groups=input_size)
        self.point_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1)
        self.act = nn.ReLU()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, out_size)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, x, hidden=None):
        x = x.permute(0, 2, 1)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        lstm_out, hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        gen = self.linear(last_out)
        cls = self.classifier(last_out)
        return gen, cls


# ==============================================================================
# === 新增模型: Generator_bigru (双向GRU) ===
# ==============================================================================
class Generator_bigru(nn.Module):
    def __init__(self, input_size, out_size, hidden_dim=512, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 关键改动: bidirectional=True
        self.gru = nn.GRU(input_size, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

        # 关键改动: 输入维度是 hidden_dim * 2，因为是双向的
        self.linear_1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear_3 = nn.Linear(hidden_dim // 2, out_size)
        self.dropout = nn.Dropout(0.2)

        # 分类器的输入维度也是 hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        # 从双向GRU的输出中提取最后一个时间步的特征
        # gru_out 形状: (batch, seq_len, num_directions * hidden_size)
        # 我们需要前向的最后一个和后向的第一个时间步的隐藏状态
        # 一个简单有效的方法是直接用最后一个时间步的拼接输出
        last_feature = self.dropout(gru_out[:, -1, :])

        # 回归预测
        gen = self.linear_1(last_feature)
        gen = self.linear_2(gen)
        gen = self.linear_3(gen)

        # 分类预测
        cls = self.classifier(last_feature)

        return gen, cls


# ==============================================================================
# === 新增模型: Generator_bilstm (双向LSTM) ===
# ==============================================================================
class Generator_bilstm(nn.Module):
    def __init__(self, input_size, out_size, hidden_size=512, num_layers=2, dropout=0.1):
        super().__init__()
        # 关键改动: bidirectional=True
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)

        # 关键改动: 输入维度是 hidden_size * 2
        self.linear = nn.Linear(hidden_size * 2, out_size)
        self.classifier = nn.Linear(hidden_size * 2, 3)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)

        # 提取最后一个时间步的拼接输出
        last_out = lstm_out[:, -1, :]

        # 回归和分类
        gen = self.linear(last_out)
        cls = self.classifier(last_out)

        return gen, cls


class Generator_transformer(nn.Module):
    def __init__(self, input_dim, feature_size=512, num_layers=2, num_heads=16, dropout=0.1, output_len=1):
        super().__init__()
        self.feature_size = feature_size
        self.output_len = output_len
        self.input_projection = nn.Linear(input_dim, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, output_len)
        self.classifier = nn.Linear(feature_size, 3)
        self._init_weights()
        self.src_mask = None

    def _init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.bias)
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, seq_len, _ = src.size()
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        last_feature = output[:, -1, :]
        gen = self.decoder(last_feature)
        cls = self.classifier(last_feature)
        return gen, cls


class Generator_transformer_deep(nn.Module):
    def __init__(self, input_dim, feature_size=256, num_layers=2, num_heads=16, dropout=0.1, output_len=1):
        super().__init__()
        self.feature_size = feature_size
        self.output_len = output_len
        self.input_projection = nn.Linear(input_dim, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, output_len)
        self.classifier = nn.Linear(feature_size, 3)
        self._init_weights()
        self.src_mask = None

    def _init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.bias)
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, seq_len, _ = src.size()
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        last_feature = output[:, -1, :]
        gen = self.decoder(last_feature)
        cls = self.classifier(last_feature)
        return gen, cls


class Generator_rnn(nn.Module):
    def __init__(self, input_size):
        super(Generator_rnn, self).__init__()
        self.rnn_1 = nn.RNN(input_size, 1024, batch_first=True)
        self.rnn_2 = nn.RNN(1024, 512, batch_first=True)
        self.rnn_3 = nn.RNN(512, 256, batch_first=True)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        device = x.device
        h0_1 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1, _ = self.rnn_1(x, h0_1)
        out_1 = self.dropout(out_1)
        h0_2 = torch.zeros(1, x.size(0), 512).to(device)
        out_2, _ = self.rnn_2(out_1, h0_2)
        out_2 = self.dropout(out_2)
        h0_3 = torch.zeros(1, x.size(0), 256).to(device)
        out_3, _ = self.rnn_3(out_2, h0_3)
        out_3 = self.dropout(out_3)
        last_feature = out_3[:, -1, :]
        gen = self.linear_1(last_feature)
        gen = self.linear_2(gen)
        gen = self.linear_3(gen)
        cls = self.classifier(last_feature)
        return gen, cls


class Discriminator3(nn.Module):
    def __init__(self, input_dim, out_size, num_cls):
        super().__init__()
        self.label_embedding = nn.Embedding(num_cls, 32)
        self.conv_x = nn.Conv1d(1, 32, kernel_size=3, padding='same')
        self.conv_label = nn.Conv1d(32, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(32 + 32, 64, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same')
        self.linear1 = nn.Linear(128, 220)
        self.batch1 = nn.BatchNorm1d(220)
        self.linear2 = nn.Linear(220, 220)
        self.batch2 = nn.BatchNorm1d(220)
        self.linear3 = nn.Linear(220, out_size)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, label_indices):
        if label_indices.ndim == x.ndim:
            label_indices = label_indices.squeeze(-1)
        label_indices = label_indices.long()
        x_seq = x.permute(0, 2, 1)
        x_feat = self.leaky(self.conv_x(x_seq))
        embedded = self.label_embedding(label_indices)
        embedded = embedded.permute(0, 2, 1)
        label_feat = self.leaky(self.conv_label(embedded))
        combined = torch.cat([x_feat, label_feat], dim=1)
        conv2_out = self.leaky(self.conv2(combined))
        conv3_out = self.leaky(self.conv3(conv2_out))
        pooled = torch.mean(conv3_out, dim=2)
        out = self.linear1(pooled)
        out = self.batch1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.batch2(out)
        out = self.relu(out)
        out = self.sigmoid(self.linear3(out))
        return out