import os
import cv2
from torch.utils import data
import numpy as np
#from tools import normalization, extractClassname, extractFilename
#from  sklearn.decomposition import PCA


def extractFilename(filePath = None):
    (filePath, tempFilename) = os.path.split(filePath)
    (fileName, extension) = os.path.splitext(tempFilename)

    return fileName

def extractClassname(name):
    return "".join(filter(str.isalpha, name))

def normalization(img):
    # 图片的标准化
    mean = np.mean(img)
    var = np.mean(np.square(img - mean))
    norImg = (img - mean)/np.sqrt(var+0.01)
    return norImg

class ReadData(data.Dataset):
    
    def __init__(self, imgPath=None):
        self.imgPath = imgPath
        self.fileName = os.listdir(imgPath)
        self.classList = ["dujiangp", "Easonp", "Jayp", "JohnMayerp", "LinYunerp",
                     "Merkelp", "Obamap", "Putinp", "XiJinpingp", "ZhaoLiyingp"]
        
    def __getitem__(self, index):
        imgName = self.imgPath + self.fileName[index]
        #grayImg = cv2.imread(imgName).transpose(2, 0, 1)
        grayImg = cv2.cvtColor(cv2.imread(imgName), cv2.COLOR_BGR2GRAY)
        #grayImg = cv2.resize(grayImg, (224, 224))
        normImg = np.reshape(normalization(grayImg), (-1, 227, 227))
        tmp_filename = extractFilename(self.fileName[index])
        train_class = extractClassname(tmp_filename)
        #print(train_class)
        Y = None
            # 读入数据集的标签
        if train_class == self.classList[0]:
            Y=0
        elif train_class == self.classList[1]:
            Y=1
        elif train_class == self.classList[2]:
            Y=2
        elif train_class == self.classList[3]:
            Y=3
        elif train_class == self.classList[4]:
            Y=4
        elif train_class == self.classList[5]:
            Y=5
        elif train_class == self.classList[6]:
            Y=6
        elif train_class == self.classList[7]:
            Y=7
        elif train_class == self.classList[8]:
            Y=8
        elif train_class == self.classList[9]:
            Y=9

        #print(np.shape(normImg), Y)
        return normImg, Y
    
    def __len__(self):
        
        return len(os.listdir(self.imgPath))


if __name__ == '__main__':
    trainSet = ReadData(imgPath='../../samples/testImg/train/')
    testSet = ReadData(imgPath='../../samples/testImg/train/')
    trainloader = data.DataLoader(trainSet,batch_size=8,shuffle=True,num_workers=1)
    for step, (data, label) in enumerate(trainloader):
        print(data)
        print(label)