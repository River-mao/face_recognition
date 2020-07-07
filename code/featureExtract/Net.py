import torch
import numpy as np
import torch.nn as nn

num_classes = 10

class AlexNet(nn.Module):

    def __init__(self, num_classes=num_classes, if_features=False):

        self.if_features = if_features
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  #(227-11+4)/4+1=56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (56-3)/2 = 26 #(56-3)/2+1=27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # (27-5+4)+1=27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (27-3)/2+1=13
            nn.Conv2d(192, 384, kernel_size=3, padding=1), #13-3+2+1 = 13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 13-3+2+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 13-3+2+1=13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (13-3)/2+1=6
        )

        self.fc1 = nn.Sequential(nn.Linear(256*6*6, 4096),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(4096, 1024),
                                 nn.ReLU(inplace=True))

        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        bs = np.shape(x)[0]
        # bs, c, w, w
        output = self.features(x).reshape(bs, -1)
        output = self.fc2(self.fc1(output))
        if self.if_features:
            return output
        else:
            output = self.fc3(output)
            return output

if __name__ == '__main__':
    # Example
    net = AlexNet()
    print(net)
