import torch
import cv2
import numpy as np
from .Net import AlexNet
from .dataset import ReadData

def extract(input=None, if_predict=True, if_train=False, paramPath=None):
    net = AlexNet(if_features=True)
    net.load_state_dict(torch.load(paramPath))

    if if_train:
        set = ReadData(imgPath=input)
        setloader = torch.utils.data.DataLoader(set, batch_size=1, shuffle=True, num_workers=1)
        features = []
        labels = []

        for step, (input, label) in enumerate(setloader):
            input = torch.tensor(input, dtype=torch.float)
            score = net(input)
            features.append(score.squeeze(0).data.numpy())
            labels.append(label.squeeze(0).data.numpy())

        print("feature type:{}, label type:{}".format(type(features), type(labels)))
        print("feature shape:{}, label shape:{}".format(np.shape(features), np.shape(labels)))
        return features, labels

    if if_predict:
        input = torch.tensor(input, dtype=torch.float).reshape(1, 1, 227, 227)
        print("input type:", type(input))
        feature = net(input)

        return feature

