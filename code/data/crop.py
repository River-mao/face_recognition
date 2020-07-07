# Author: River Mao
# 功能： 图片裁剪，生成正负样本集合

# 库函数初始化
import cv2
import os
import numpy as np
from utils import normalization, imgShow
from overlap import cal_overlap

# 每张图片候选负样本图片的数目
selectNum = 2
# 重叠区域小于该值选择为负例
overlapThreP = 0.9
overlapThreN = 0.4

dsize = (227, 227)

# 数据集路径
datasetPath = "../../data/dataset/"

# 初始化保存路径
# 依据裁剪的参数来命名文件夹
posPath = '../../data/samples/size_'+str(dsize[0])\
          +'_pt_' + str(int(100*overlapThreP)) \
          +'_nt_' + str(int(100*overlapThreN)) \
          + '_num_'+str(selectNum)+'/pos'

negPath = '../../data/samples/size_'+str(dsize[0])\
          +'_pt_' + str(int(100*overlapThreP)) \
          +'_nt_' + str(int(100*overlapThreN)) \
          + '_num_'+str(selectNum)+'/neg'

def randomArea(spt, ept, ledges):
    '''
    输入：
        spt: strat point, (y, x)，
        ept: end point, (y, x)
        ROI区域的左上角的点落在以sptw为左上角ept为右下角的矩形区域中
        ledges: 方框的边长
    输出：
        ROI区域的参数
        ....................
        ..........         .
        ..............     .
        ..........   .     .
        .     ........     .
        .                  .
        ....................
    '''

    # 随机选取坐标点
    yStrat = np.random.randint(spt[0], ept[0])
    xStart = np.random.randint(spt[1], ept[1])
    ROIP = [yStrat, xStart, ledges, ledges]

    return ROIP

if __name__ == '__main__':

    if not os.path.exists(posPath):
        os.makedirs(posPath)

    if not os.path.exists(negPath):
        os.makedirs(negPath)

    # 读取所有人物的人名
    classList = os.listdir(datasetPath)
    # 根据人物依次对图像进行裁剪
    for i in range(len(classList)):
        # 当前人物的人名
        className = classList[i]
        # 当前人物的图像路径
        classPath = datasetPath + className + '/'
        # 当前人物的标签路径
        labelPath = classPath + className + '.txt'
        labelList = []

        # 获取当前图片的标签（路径， 坐标参数）
        # labelList 每一行代表一张图片标签
        with open(labelPath, 'r') as f:
            for line in f.readlines():
                labelList.append(line.strip('\n').split(','))

        imgindex = 0

        for label in labelList:
            imgindex +=1

            # 标签0号位置是图片名
            imgPath = datasetPath + label[0]

            # 读取图片，并将图片转化为灰度图格式
            img = cv2.imread(imgPath)
            grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #grayImg = cv2.imread(imgPath)
            # 获取图片的大小
            imgSize = np.shape(grayImg)

            # 以标签1.2倍裁剪出待选择的正例区域，并在待选区域中随机裁剪至所需数目.
            # 将裁剪后的图片统一到dsize维度，保存到正例文件夹中。
            # 其中， dsize在该文件的开头超参区设置
            # 因为标签框为正方形框，所以记录label[3]或者label[4]其中之一就可以
            # 矩形参数在标签中以字符形式储存，先完成数据的转换
            # lEdges 原标签中的ROI边长, cEdges 拓宽之后的区域， dEdges 两边长差
            lEdges = int(label[3])
            cEdges = int(1.3 * lEdges)
            dEdges = cEdges - lEdges
            # label: img_path, x, y, edges, edges
            pt0 = [int(label[2]), int(label[1])]
            # 正样区左上角点可选区域
            pt1 = [int(pt0[0])-int(0.5*dEdges), int(pt0[1])-int(0.5*dEdges)]
            pt2 = [int(pt0[0])+int(0.5*dEdges), int(pt0[1])+int(0.5*dEdges)]
            pArea = [pt1, pt2]
            # 负样区左上角点可选区域
            pt3 = [0, 0]
            pt4 = [imgSize[0]-lEdges, imgSize[1]-lEdges]
            nArea = [pt3, pt4]

            # 候选正样例区
            #posImgArea = grayImg[pt1[0]:pt1[0]+cEdges, pt1[1]:pt1[1]+cEdges]
            lROI = [pt0[0], pt0[1], lEdges, lEdges]


            # 图片计数
            neg_index = 0
            pos_index = 0

            while pos_index != selectNum:
                pROI = randomArea(pArea[0], pArea[1], lEdges)
                overlapRatio = cal_overlap(pROI, lROI, grayImg)
                if overlapRatio > overlapThreP:
                    pos_index += 1
                    posImg = grayImg[pROI[0]:pROI[0] + pROI[2], pROI[1]:pROI[1] + pROI[2]]
                    posImg = cv2.resize(posImg, dsize)
                    #imgShow(img =posImg)
                    cv2.imwrite(posPath + '/' + className + '_' + str(imgindex) + '_' + str(pos_index) + 'p' + '.png', posImg)


            while neg_index != selectNum:
                nROI = randomArea(nArea[0], nArea[1], lEdges)
                overlapRatio = cal_overlap(nROI, lROI, grayImg)
                if overlapRatio < overlapThreN:
                    neg_index +=1
                    negImg = grayImg[nROI[0]:nROI[0] + nROI[2], nROI[1]:nROI[1] + nROI[2]]
                    negImg = cv2.resize(negImg, dsize)
                    #imgShow(img=negImg)
                    cv2.imwrite(negPath + '/' + className + '_' + str(imgindex) + '_'+ str(neg_index) + 'n' +'.png', negImg)

