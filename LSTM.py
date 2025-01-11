import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import time


# LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        # 隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        # 细胞状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out


# 引入注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, hidden_size)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        out = torch.sum(attention_weights * x, dim=1)
        return out


# 双向LSTM + 注意力机制
class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers=1):
        super(BiLSTMAttention, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_layer_size * 2)  # 输出维度为hidden_size*2
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_layer_size).to(x.device)  # 双向LSTM的初始状态
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_layer_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        attention_out = self.attention(lstm_out)
        out = self.linear(attention_out)
        return out


# CNN + LSTM混合模型
class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers=1):
        super(CNNLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_out = self.conv1d(x)
        conv_out = conv_out.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        lstm_out, _ = self.lstm(conv_out, (h0, c0))
        out = self.linear(lstm_out[:, -1, :])
        return out


# 时间卷积块
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2  # 根据卷积核大小和膨胀系数计算填充量，确保卷积后输出大小保持一致
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        if self.downsample is not None:  # 降采样调整形状
            residual = self.downsample(x)
        return self.relu(out + residual)


# 时间卷积网络
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)  # 连接卷积块
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out.permute(0, 2, 1)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.linear(out)
        return out


# 数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_col, seq_length=10):
        self.data = data
        self.target_col = target_col
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx + self.seq_length].drop(columns=[self.target_col]).values
        y = self.data.iloc[idx + self.seq_length][self.target_col]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train(model_name, lr=0.005, hidden_layer_size=10, num_layers=1, epochs=20):
    hidden_layer_size = int(hidden_layer_size)
    num_layers = int(num_layers)

    # 加载处理后的数据集
    train_data = pd.read_csv('./data/train_dataset.csv')
    val_data = pd.read_csv('./data/val_dataset.csv')
    test_data = pd.read_csv('./data/test_dataset.csv')

    # 时间格式转换
    train_data['date_time'] = pd.to_datetime(train_data['date_time'])
    val_data['date_time'] = pd.to_datetime(val_data['date_time'])
    test_data['date_time'] = pd.to_datetime(test_data['date_time'])
    x = len(test_data['date_time'])

    num_cols = ['temp', 'rain_1h', 'clouds_all', 'traffic_volume']
    bool_cols = train_data.columns.difference(num_cols).difference(['date_time'])

    # 编码
    train_data[bool_cols] = train_data[bool_cols].map(lambda x: 1 if x == 'TRUE' else 0)
    val_data[bool_cols] = val_data[bool_cols].map(lambda x: 1 if x == 'TRUE' else 0)
    test_data[bool_cols] = test_data[bool_cols].map(lambda x: 1 if x == 'TRUE' else 0)

    train_data.drop(columns=['date_time'], inplace=True)
    val_data.drop(columns=['date_time'], inplace=True)
    test_data.drop(columns=['date_time'], inplace=True)

    # 创建数据集和数据加载器
    seq_length = 10
    target_col = 'traffic_volume'
    train_dataset = TimeSeriesDataset(train_data, target_col, seq_length)
    val_dataset = TimeSeriesDataset(val_data, target_col, seq_length)
    test_dataset = TimeSeriesDataset(test_data, target_col, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = train_data.shape[1] - 1
    output_size = 1
    num_channels = [64, 128, 256]

    # 选择模型
    if model_name == 'bilstm':
        model = BiLSTMAttention(input_size, hidden_layer_size, output_size, num_layers)
        model_path = './model/bilstm.pt'
    elif model_name == 'cnnlstm':
        model = CNNLSTM(input_size, hidden_layer_size, output_size, num_layers)
        model_path = './model/cnnlstm.pt'
    elif model_name == 'tcn':
        model = TCN(input_size, output_size, num_channels)
        model_path = './model/tcn.pt'
    elif model_name == 'lstm':
        model = LSTM(input_size, hidden_layer_size, output_size, num_layers)
        model_path = './model/lstm.pt'
    else:
        model = LSTM(input_size, hidden_layer_size, output_size, num_layers)
        model_path = './model/lstm.pt'

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 动态学习率调整
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    num_epochs = epochs
    start_time = time.time()  # 记录训练开始时间

    # 记录每一代的训练损失和验证损失
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()  # 记录每个epoch的开始时间

        # 训练阶段
        epoch_train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = model(x_batch)
                val_loss += criterion(y_pred, y_batch.unsqueeze(1)).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)  # 动态调整学习率

        # 每5代输出预期训练完成时间
        if (epoch + 1) % 5 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = num_epochs - (epoch + 1)
            expected_remaining_time = remaining_epochs * avg_time_per_epoch
            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {epoch_train_loss:.4f}, Validation Loss: {val_loss: .4f}, '
                  f'Expected Time to Finish: {expected_remaining_time / 60: .2f} minutes')

    # 保存模型参数
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig(f'./image/{model_name}_loss.jpg', dpi=300)

    # 测试和预测
    model.eval()
    predictions = []
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            y_pred = model(x_batch)
            test_loss += criterion(y_pred, y_batch.unsqueeze(1)).item()

            # 记录预测值以进行可视化
            predictions.extend(y_pred.cpu().squeeze(1).tolist())

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

    # 预测结果逆变换
    predictions = np.array(predictions).reshape(-1, 1)
    traffic_volume_scaler = StandardScaler()
    traffic_volume_scaler.fit_transform(test_data[['traffic_volume']])
    predictions = traffic_volume_scaler.inverse_transform(predictions).flatten()

    # 将预测添加到test_data
    test_data['traffic_volume_prediction'] = np.nan
    test_data.loc[seq_length:, 'traffic_volume_prediction'] = predictions

    # 可视化 predictions vs true traffic volume
    plt.figure(figsize=(20, 8))
    plt.plot(list(range(1, x + 1)), test_data['traffic_volume'], label='True Traffic Volume', color='blue')
    plt.plot(list(range(seq_length, x)), test_data['traffic_volume_prediction'][seq_length:],
             label='Predicted Traffic Volume', color='red')
    plt.xlabel('Time')
    plt.ylabel('Traffic Volume')
    plt.title('True vs Predicted Traffic Volume')
    plt.legend()
    plt.savefig(f'./image/{model_name}_predict.jpg', dpi=300)


if __name__ == '__main__':
    # 四个模型依次进行训练比对
    model_ist = ['lstm', 'bilstm', 'cnnlstm', 'tcn']
    for model in model_ist:
        train(model_name=model, epochs=200)
    plt.show()
