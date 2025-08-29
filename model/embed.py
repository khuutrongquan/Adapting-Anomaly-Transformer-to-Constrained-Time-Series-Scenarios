import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 activation,
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        if activation =='relu':
            self.act = nn.ReLU()
        elif activation == 'leak':
            self.act = nn.LeakyReLU()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)


        self.net = nn.Sequential(self.conv1, self.chomp1, self.act, self.dropout1,
                                 self.conv2, self.chomp2, self.act, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)
    
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels=[32, 64, 128], kernel_size=3, dropout=0.0, activation='relu'):
        super(TemporalConvNet, self).__init__()
        self.n_inputs = num_inputs
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, activation=activation)]

        self.network = nn.Sequential(*layers)
        self.flat = nn.Flatten()
    
    def forward(self, x):
        x = self.network(x)
        return x

class TCNEmbedding(nn.Module):
    def __init__(self, c_in, d_model, num_channels=[32, 64, 128], kernel_size=3, dropout=0.0):
        super(TCNEmbedding, self).__init__()
        self.tcn = TemporalConvNet(c_in, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], d_model)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y1 = self.tcn(x)         # output: [B, C_out, L]
        y1 = y1.transpose(1, 2)  # → [B, L, C_out]
        return self.linear(y1)   # → [B, L, d_model]

class ConvolutionalEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ConvolutionalEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tcn = TCNEmbedding(c_in, d_model)
        self.cnn = ConvolutionalEmbedding(c_in, d_model)

    def forward(self, x):
        x = self.tcn(x) + self.cnn(x)
        return x

class DoubleSequentialConvolutionalEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(DoubleSequentialConvolutionalEmbedding, self).__init__()
        self.tokenConv1 = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, bias=False)
        self.tokenConv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=3, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv1(x.permute(0, 2, 1))
        x = F.leaky_relu(x)
        x = self.tokenConv2(x).transpose(1, 2)
        return x

class DoubleSequentialTCNEmbedding(nn.Module):
    def __init__(self, c_in, d_model, num_channels=[32, 64, 128], kernel_size=3, dropout=0.0):
        super(DoubleSequentialTCNEmbedding, self).__init__()
        self.tcn1 = TemporalConvNet(c_in, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn2 = TemporalConvNet(num_channels[-1], num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], d_model)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn1(x)                          # [B, C_out, L]
        out = self.tcn2(out)                        # [B, C_out, L]
        out = out.transpose(1, 2)                   # [B, L, C_out]
        return self.linear(out)                     # [B, L, d_model]

class DoubleParallelConvolutionalEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(DoubleParallelConvolutionalEmbedding, self).__init__()
        self.tokenConv1 = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, bias=False)
        self.tokenConv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=3, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x1 = self.tokenConv1(x.permute(0, 2, 1))    # [B, d_model, L]
        x2 = self.tokenConv1(x.permute(0, 2, 1))    # [B, d_model, L]
        out = x1 + x2                               # [B, d_model, L]
        return out.permute(0, 2, 1)                 # [B, L, d_model]
    
class DoubleParallelTCNEmbedding(nn.Module):
    def __init__(self, c_in, d_model, num_channels=[32, 64, 128], kernel_size=3, dropout=0.0):
        super(DoubleParallelTCNEmbedding, self).__init__()
        self.tcn1 = TemporalConvNet(c_in, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn2 = TemporalConvNet(c_in, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], d_model)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y1 = self.tcn1(x)                           # output: [B, C_out, L]
        y2 = self.tcn2(x)                           # output: [B, C_out, L]
        y = y1 + y2
        y = y.transpose(1, 2)                       # → [B, L, C_out]
        return self.linear(y)                       # → [B, L, d_model]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.value_embedding = DoubleParallelTCNEmbedding(c_in=c_in, d_model=d_model)
        # self.value_embedding = DoubleParallelConvolutionalEmbedding(c_in=c_in, d_model=d_model)
        # self.value_embedding = DoubleSequentialTCNEmbedding(c_in=c_in, d_model=d_model)
        # self.value_embedding = DoubleSequentialConvolutionalEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
