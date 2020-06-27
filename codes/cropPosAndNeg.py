# Author: River Mao

import cv2
import os
import numpy as np
from cal_overlap import cal_overlap


# 每张图片候选负样本图片的数目
selectNum = 10
# 重叠区域小于该值选择为负例
overlapThre = 0.4
# 输出形状大小
# 因为标签人脸框为正方形
# 这里也尽量保持正方形，避免图像比例失调
dsize = (24, 24)

# 正负样本集文件夹
# 文件夹中包含：
# 1 切割后图片的形状大小
# 2 重叠阈值
# 3 每张图片候选负样本图片的数目

posPath = '../samples/size_'+str(dsize[0])+'_t_'+str(int(100*overlapThre))+'_num_'+str(selectNum)+'/pos'
negPath = '../samples/size_'+str(dsize[0])+'_t_'+str(int(100*overlapThre))+'_num_'+str(selectNum)+'/neg'

def randomArea(imgSize, edges):
    startArea = imgSize[0]-edges, imgSize[1]-edges
    yStrat = np.random.randint(startArea[0])
    xStart = np.random.randint(startArea[1])
    ROIP = [yStrat, xStart, edges, edges]

    return ROIP

def imgShow(winname, img):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    datasetPath = '../dataset/'
    classList = os.listdir(datasetPath)

    for i in range(len(classList)):
        className = classList[i]
        classPath = datasetPath + className + '/'
        labelPath = classPath + className + '.txt'
        labelList = []

        with open(labelPath, 'r') as f:
            for line in f.readlines():
                labelList.append(line.strip('\n').split(','))

        # 每一行代表一张图片标签
        # 取标签
        imgindex = 0
        for label in labelList:
            imgindex +=1
            # 标签0号位置是图片名
            imgPath = datasetPath + label[0]
            img = cv2.imread(imgPath)
            grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgSize = np.shape(grayImg)

            # 因为标签框为正方形框，所以记录label[3]或者label[4]其中之一就可以
            edges = int(label[3])
            ROI1P = [int(label[2]), int(label[1]), edges, edges]
            # 截取正样本区
            posImg = grayImg[ROI1P[0]:ROI1P[0]+edges, ROI1P[1]:ROI1P[1]+edges]
            posImg = cv2.resize(posImg, dsize)

            #imgShow("posImg", posImg)
            if not os.path.exists(posPath):
                os.makedirs(posPath)

            cv2.imwrite(posPath+ '/'+ className + str(imgindex)+'.png', posImg)

            # 截取负样本区
            sample_index = 0
            for k in range(selectNum):
                ROI2P = randomArea(imgSize, edges)
                overlapRatio = cal_overlap(ROI1P, ROI2P, grayImg)
                if overlapRatio < overlapThre:
                    sample_index += 1
                    negImg = grayImg[ROI2P[0]:ROI2P[0]+edges,ROI2P[1]:ROI2P[1]+edges]
                    negImg = cv2.resize(negImg, dsize)
                    if not os.path.exists(negPath):
                        os.makedirs(negPath)

                    cv2.imwrite(negPath + '/' + className + str(imgindex)+'_'+str(sample_index)+'.png', negImg)

                    #imgShow("negImg", negImg)
