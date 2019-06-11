import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv=3):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, conv, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)


        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x

class Kaggle2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (7, 3), 1, 1),
            #nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #x = F.max_pool2d(x, [1, 3])

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 7), 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 10), 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, (7, 1), 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, [1, 3])
        x = self.conv2(x)
        x = F.max_pool2d(x, [1, 4])
        x = self.conv3(x)
        x = self.conv4(x)
        ##todo
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)        #print(x.size())
        x = self.fc(x.view(-1, 512))
        return x


class WavToLetterConv(nn.Module):
    def __init__(self, in_channels, out_channels, kw=7):
        super().__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kw),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_1(x)
        return x

class MLP_(nn.Module):
    def __init__(self, sincnet_options:dict, num_classes:int):
        super().__init__()
        self.sincnet = SincNet(sincnet_options)

        self.fc_1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(45600, 2048),
            nn.PReLU()
        )
        
        self.fc_2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.PReLU()
        )

        self.fc_3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.PReLU()
        )

        self.fc_last = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        
        self.conv_1 = WavToLetterConv(60, 250)
        self.conv_2 = WavToLetterConv(250, 250)
        self.conv_3 = WavToLetterConv(250, 250)
        self.conv_4 = WavToLetterConv(250, 250)
        self.conv_5 = WavToLetterConv(250, 250)
        self.conv_6 = WavToLetterConv(250, 250)
        self.conv_7 = WavToLetterConv(250, 250)
        self.conv_8 = WavToLetterConv(250, 2000, kw=32)
        self.conv_9 = WavToLetterConv(2000, 2000, kw=1)
        self.conv_10 = WavToLetterConv(2000, 40, kw=1)
        

        
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.sincnet(x)
        #x = self.pool(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.conv_8(x)
        x = self.conv_9(x)
        x = self.conv_10(x)
        
        x = x.view(x.size()[0], -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.fc_last(x)
        #x = self.sig(x)
        return x