import torch

input_size = 256
output_size = 15

class NetWork(torch.nn.Module):

    def __init__(self, input_size = 256, output_size = 15):
        super(NetWork, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, 512)
        self.relu1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(256, 64)
        self.relu3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Linear(64, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input=None):
        X = self.layer1(input)
        X = self.relu1(X)
        X = self.layer2(X)
        X = self.relu2(X)
        X = self.layer3(X)
        X = self.relu3(X)
        X = self.layer4(X)
        Y = self.sigmoid(X)

        return Y