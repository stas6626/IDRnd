import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyJasper(nn.Module):
    def __init__(self, in_channels=64):
        super(TinyJasper, self).__init__()

        self.first_layer = C(256, 256, 11, stride=2, dropout_rate=0.2)

        self.B1 = nn.Sequential(
            C(256, 256, 11, dropout_rate=0.2),
            C(256, 256, 11, dropout_rate=0.2),
            C(256, 256, 11, dropout_rate=0.2),
        )

        self.B2 = nn.Sequential(
            C(256, 384, 13, dropout_rate=0.2),
            C(384, 384, 13, dropout_rate=0.2),
            C(384, 384, 13, dropout_rate=0.2),
        )
        self.r2 = nn.Conv1d(256, 384, 1)

        self.B3 = nn.Sequential(
            C(384, 512, 17, dropout_rate=0.2),
            C(512, 512, 17, dropout_rate=0.2),
            C(512, 512, 17, dropout_rate=0.2),
        )
        self.r3 = nn.Conv1d(384, 512, 1)

        self.B4 = nn.Sequential(
            C(512, 640, 21, dropout_rate=0.3),
            # C(640, 640, 21, dropout_rate=0.3),
            # C(640, 640, 21, dropout_rate=0.3),
        )
        self.B5 = nn.Sequential(
            C(640, 768, 25, dropout_rate=0.3),
            # C(768, 768, 25, dropout_rate=0.3),
            # C(768, 768, 25, dropout_rate=0.3),
        )
        self.r4_5 = nn.Conv1d(512, 768, 1)

        self.last_layer = nn.Sequential(
            C(768, 896, 29, dropout_rate=0.4, dilation=2),
            C(896, 1024, 1, dropout_rate=0.4),
            C(1024, 80, 1, dropout_rate=0.4),
            C(80, 4, 1, dropout_rate=0.4),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 80),
        )

    def forward(self, x):
        y = self.first_layer(x)

        y = self.B1(y) + y
        y = self.B2(y) + self.r2(y)
        y = self.B3(y) + self.r3(y)
        y = self.B5(self.B4(y)) + self.r4_5(y)

        y = self.last_layer(y)

        y = y.view(-1, 1024)
        y = self.fc(y)
        return y


class C(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        activation="relu",
        dropout_rate=0.0,
    ):
        """1D Convolution with the batch normalization and RELU."""
        super(C, self).__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate

        assert 1 <= stride <= 2
        if dilation > 1:
            assert stride == 1
            padding = (kernel_size - 1) * dilation // 2
        else:
            padding = (kernel_size - stride + 1) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        nn.init.xavier_uniform_(self.conv.weight, nn.init.calculate_gain("relu"))

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)

        if self.activation == "relu":
            y = F.relu(y, inplace=True)
            # OpenSeq2Seq uses max clamping instead of gradient clipping
            # y = torch.clamp(y, min=0.0, max=20.0)  # like RELU but clamp at 20.0

        if self.dropout_rate > 0:
            y = F.dropout(y, p=self.dropout_rate, training=self.training, inplace=False)
        return y
