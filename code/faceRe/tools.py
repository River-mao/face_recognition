import cv2
import os
import numpy as np
import pickle


def normalization(img):
    # 图片的标准化
    mean = np.mean(img)
    var = np.mean(np.square(img - mean))
    norImg = (img - mean)/np.sqrt(var+0.01)
    return norImg


def save_dir(dir, name, DirPath):
    # 保存字典
    if not os.path.exists(DirPath):
        os.mkdir(DirPath)

    with open(DirPath +'/' + name + '.pkl', 'wb') as f:
        pickle.dump(dir, f, pickle.HIGHEST_PROTOCOL)

def load_dir(dirPath):
    # 读取字典
    with open(dirPath, 'rb') as f:
        return pickle.load(f)

def imgShow(winname = 'show', img= None):
    # 显示图片
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def readImgCollection(posPath, negPath):

    posDir = os.listdir(posPath)
    negDir = os.listdir(negPath)
    totalDir = {}

    for key in posDir:
        key = posPath + key
        totalDir[key] = 1
    for key in negDir:
        key = negPath + key
        totalDir[key] = -1

    return  totalDir

def select(dir, start, selectNum):
    keyList = []
    selectDir = {}
    for key in dir.keys():
        keyList.append(key)
    for i in range(selectNum):
        key = keyList[start + i]
        selectDir[i] = dir[key]

    return selectDir

def extractFilename(filePath = None):
    (filePath, tempFilename) = os.path.split(filePath)
    (fileName, extension) = os.path.splitext(tempFilename)

    return fileName

def extractClassname(name):
    return "".join(filter(str.isalpha, name))