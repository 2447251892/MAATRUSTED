import torch
import torch.nn as nn
import math

class Generator_gru(nn.Module):
    def __init__(self, input_size, out_size, hidden_dim = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)  # Keep only one GRU layer, hidden units 256
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear_2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.linear_3 = nn.Linear(hidden_dim//4, out_size)
        self.dropout = nn.Dropout(0.2)

        # Add classification head, input dimension 256, output 3 classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim//2, 3)
        )

    def forward(self, x):
        device = x.device
        # Initialize GRU hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_dim, device=device)
        # Pass through GRU layer
        out, _ = self.gru(x, h0)
        # Take the output of the last time step in the sequence, and apply dropout
        last_feature = self.dropout(out[:, -1, :])

        # Original output (e.g., for generation or regression task)
        gen = self.linear_1(last_feature)
        gen = self.linear_2(gen)
        gen = self.linear_3(gen)

        # Classification output
        cls = self.classifier(last_feature)

        return gen, cls


class Generator_lstm(nn.Module):
    def __init__(self, input_size, out_size, hidden_size=128, num_layers=2, dropout=0.1):
        """
        Args:
            input_size (int): Number of input features
            out_size (int): Output target dimension (e.g., for generating regression results)
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers, default 1 (reduce computation)
            dropout (float): Internal dropout rate for LSTM, default 0.1
        """
        super().__init__()
        # Use depthwise separable convolution: perform depthwise first, then pointwise transformation
        self.depth_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size,
                                    kernel_size=3, padding=1, groups=input_size)
        self.point_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1)
        self.act = nn.ReLU()

        # LSTM part: input channels are (input_size * 4)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        # Directly use the output of the last time step for linear mapping
        self.linear = nn.Linear(hidden_size, out_size)

        self.classifier = nn.Linear(hidden_size, 3)


    def forward(self, x, hidden=None):
        """
        Args:
            x (torch.Tensor): Input, shape (batch_size, seq_len, input_size)
            hidden: Optional initial LSTM state
        Returns:
            tuple: (gen, cls)
                gen (torch.Tensor): Regression output, shape (batch_size, out_size)
                cls (torch.Tensor): Classification output, shape (batch_size, 3)
        """
        # Adjust dimensions: transpose input from (B, T, F) to (B, F, T) for Conv1d
        x = x.permute(0, 2, 1)  # (B, input_size, T)
        # Depthwise convolution
        x = self.depth_conv(x)
        # Pointwise convolution
        x = self.point_conv(x)
        x = self.act(x)
        # Transpose back to (B, T, F')
        x = x.permute(0, 2, 1)
        # LSTM forward pass: use the last time step state as output here
        lstm_out, hidden = self.lstm(x, hidden)
        # Directly take the last time step output as feature (avoid extra pooling)
        last_out = lstm_out[:, -1, :]
        gen = self.linear(last_out)
        cls = self.classifier(last_out)

        return gen, cls

# Positional Encoder
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        """
        model_dim: Dimension of the model's feature vector
        max_len: Maximum supported sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, model_dim)

        # Position index
        positions = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]

        # Dimension index, scaled using exponential function
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))

        # Use sin for even positions, cos for odd positions
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension: [1, max_len, model_dim]

    def forward(self, x):
        """
        x: Input features [batch_size, seq_len, model_dim]
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)  # Only take positional information for the corresponding length


class Generator_transformer(nn.Module):
    def __init__(self, input_dim, feature_size=128, num_layers=2, num_heads=8, dropout=0.1, output_len=1):
        """
        input_dim: Data feature dimension
        feature_size: Model feature dimension
        num_layers: Number of encoder layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        output_len: Predicted time step length (original task output dimension)
        """
        super().__init__()
        self.feature_size = feature_size
        self.output_len = output_len
        self.input_projection = nn.Linear(input_dim, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        # Add batch_first=True
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, output_len)  # Original task output
        # Add classification head: input feature_size, output 3 classes

        self.classifier = nn.Linear(feature_size, 3)

        self._init_weights()
        self.src_mask = None

    def _init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask=None):
        batch_size, seq_len, _ = src.size()
        src = self.input_projection(src)
        src = self.pos_encoder(src)

        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)

        output = self.transformer_encoder(src, src_mask)
        # Take the last time step as feature representation [batch_size, feature_size]
        last_feature = output[:, -1, :]

        # Original task output
        gen = self.decoder(last_feature)
        # Classification output
        cls = self.classifier(last_feature)

        return gen, cls

    def _generate_square_subsequent_mask(self, seq_len):
        # Generate upper triangular mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

# RNN Generator Model
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

    def forward(self, x):
        device = x.device # Correctly get device from input tensor
        h0_1 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1, _ = self.rnn_1(x, h0_1)
        out_1 = self.dropout(out_1)
        h0_2 = torch.zeros(1, x.size(0), 512).to(device)
        out_2, _ = self.rnn_2(out_1, h0_2)
        out_2 = self.dropout(out_2)
        h0_3 = torch.zeros(1, x.size(0), 256).to(device)
        out_3, _ = self.rnn_3(out_2, h0_3)
        out_3 = self.dropout(out_3)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out = self.linear_3(out_5)
        return out

class Discriminator3(nn.Module):
    def __init__(self, input_dim, out_size, num_cls):
        """
        input_dim: Number of features at each time step, e.g., 21
        out_size: Number of prediction values you want to output, e.g., 5
        """
        super().__init__()
        # Regression value processing branch
        self.label_embedding = nn.Embedding(num_cls, 32)
        self.conv_x = nn.Conv1d(1, 32, kernel_size=3, padding='same')
        # Label embedding processing branch
        self.conv_label = nn.Conv1d(32, 32, kernel_size=3, padding='same')

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding='same')
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
        """
        x: [B, W, 1] Regression values
        labels: [B, W] hard label (integer type)
        """
        # Process regression values
        x = x.permute(0, 2, 1)  # [B, 1, W]
        x_feat = self.leaky(self.conv_x(x))  # [B, 32, W]

        # Process label embedding
        embedded = self.label_embedding(label_indices)  # [B, W, embedding_dim]
        embedded = embedded.squeeze().permute(0, 2, 1)  # [B, embedding_dim, W]
        label_feat = self.leaky(self.conv_label(embedded))  # [B, 32, W]

        # Concatenate features
        combined = torch.cat([x_feat, label_feat], dim=1)  # [B, 64, W]
        conv2 = self.leaky(self.conv2(combined))  # [B, 64, W]
        conv3 = self.leaky(self.conv3(conv2))  # [B, 128, W]

        # Aggregate time information, take the mean
        pooled = torch.mean(conv3, dim=2)  # [B, 128]

        out = self.leaky(self.linear1(pooled))  # [B, 220]
        out = self.relu(self.linear2(out))     # [B, 220]
        out = self.sigmoid(self.linear3(out))  # [B, out_size]

        return out