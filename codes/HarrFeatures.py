# Author: River Mao
# 使用积分图 实现Harr A 特征
# 选取的模板都是比待检测区域小的正方形区域
# 模板是：“待检测区域大小”、 “模板位置”和“模板大小”三个参量的函数
# 对于这三个参数我们有以下规定：
# 1 待检测区域:
#   一般由上游任务给定， 在该任务中我们规定为（24， 24）
# 2 模板大小与模板位置
#   计划使用 Harr A Features 实现级联分类器
#   每一级分类器都是一个强分类器，该强分类器由多个弱分类器组成， 每一个分类器都是一个 Harr A 模板
#   我们设置每一个强分类器由不大于20个的弱分类器组成
#   每一个级联分类器由不大于10个的强分类器级联而成
#   因此我们需要至多200个 Harr A 模板
#   因此
#   模板大小：(18, 16, 14, 12, 10)
#   模板位置：每一个模板大小的设置对应的左上角最大可选区域中的任意一点
#   综合以上三个参数，总共有540模板待选
#   模板的位置每次通过随机数任意生成， 并且统一模板不出现两次
#   并设置一个阈值，当该模板分类的结果小于这个阈值的时候就将该模板舍弃

import cv2
import os
from tool import normalization, save_dir, readImgCollection
posPath = '../samples/size_24_t_30_90_num_3/pos/'
negPath = '../samples/size_24_t_30_90_num_3/neg/'
DirPath="../harrFeature/size_24_t_30_90_num_3"
#18,16,14,12,10,8,6
epsilonList = [0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33]

class HarrFeatures(object):

    def __init__(self, imgCollection=None, epsilonList=None):

        # 图片与阈值集合
        self.imgCollection = imgCollection
        self.epsilonList = epsilonList

        # collection的数据结构为字典
        # 每一个模板在collection以左上角的坐标作为值储存
        # 键为模板的序号， 为方便调用
        self.harrCollection18All = self.harrCollectionAll(18)
        self.harrCollection16All = self.harrCollectionAll(16)
        self.harrCollection14All = self.harrCollectionAll(14)
        self.harrCollection12All = self.harrCollectionAll(12)
        self.harrCollection10All = self.harrCollectionAll(10)
        self.harrCollection8All  = self.harrCollectionAll(8)
        self.harrCollection6All  = self.harrCollectionAll(6)

        # 满足条件的Harr分类器
        self.harrCollection18 = None
        self.harrCollection16 = None
        self.harrCollection14 = None
        self.harrCollection12 = None
        self.harrCollection10 = None
        self.harrCollection8  = None
        self.harrCollection6  = None

    def acceptedHarrCollectionAll(self):
        print(" cal the all the accepted Harr collection")
        #self.harrCollection18 = self.calAcceptedHarrCollection(self.harrCollection18All, self.epsilonList[0])
        self.harrCollection16 = self.calAcceptedHarrCollection(self.harrCollection16All, self.epsilonList[1])
        save_dir(dir=harrFeatures.harrCollection16, name="harrColl16", DirPath=DirPath)
        self.harrCollection14 = self.calAcceptedHarrCollection(self.harrCollection14All, self.epsilonList[2])
        save_dir(dir=harrFeatures.harrCollection14, name="harrColl14", DirPath=DirPath)
        self.harrCollection12 = self.calAcceptedHarrCollection(self.harrCollection12All, self.epsilonList[3])
        save_dir(dir=harrFeatures.harrCollection12, name="harrColl12", DirPath=DirPath)
        self.harrCollection10 = self.calAcceptedHarrCollection(self.harrCollection10All, self.epsilonList[4])
        save_dir(dir=harrFeatures.harrCollection10, name="harrColl10", DirPath=DirPath)
        self.harrCollection8 = self.calAcceptedHarrCollection(self.harrCollection8All, self.epsilonList[5])
        save_dir(dir=harrFeatures.harrCollection8, name="harrColl8", DirPath=DirPath)
        self.harrCollection6 = self.calAcceptedHarrCollection(self.harrCollection6All, self.epsilonList[6])
        save_dir(dir=harrFeatures.harrCollection6, name="harrColl6", DirPath=DirPath)

        print("All the accepted harr collections done!")

    def calAcceptedHarrCollection(self, harrCollection, maxEpsilon = None):
        print(" cal the one accepted Harr collection")
        '''
        input:
            imgCollection: 输入图片集合，包含所有的正负例。字典结构，键为图片名， 值为图片的label
            harrCollection: 输入harr模板的位置、大小
            maxEpsilon: 最大可允许误差，当所计算的误差大于某一个值时， 舍弃该harr模板，将那些不好的模板剔除
        output:
            acceptedCollection：符合条件的 harr模板集合
        '''

        # 符合条件的harr模板索引
        index = 0
        acceptedCollection ={}
        harrValueCollection = {}
        for items in harrCollection.items():
            harrPosition = items[1]
            harrValueList = []
            for imgName in self.imgCollection.keys():

                img = cv2.imread(imgName, cv2.COLOR_BGR2GRAY)
                SATA, SATB, SATC, SATD = self.calBlockABCD(img, harrPosition)
                harrValueCollection[imgName] = self.HarrAFeatures(SATA, SATB, SATC, SATD)
                # (Harr特征值, label, imgName)
                tmp = (harrValueCollection[imgName], self.imgCollection[imgName], imgName)
                # 每一行代表了一张图片
                harrValueList.append(tmp)

            # 通过将特征值作为每行的开头，对图片进行排序
            # 在该排序过程中，标签与图片名也同步改变
            harrValueList = sorted(harrValueList, reverse=True)

            # 通过calMinEpsilonIndex函数计算上述排好序的矩阵分类误差最小值，最小值的位置，不等号方向
            epsilon, epsilonindex, p = self.calMinEpsilonIndex(harrValueList)
            print("epsilonindex", epsilonindex)
            # 这里我们取最小误差位置的Harr特征值与前一个位置的特征值的平均值作为该Harr模板的分类阈值
            boundHarrValue = (harrValueList[epsilonindex][0] + harrValueList[epsilonindex-1][0])/2

            # boundHarrValue 是 该模板的分类阈值
            # harrPosition 是 该模板的位置信息
            # epsilon 是 该模板的最小分类误差
            # p 为方向信息
            value = [boundHarrValue, harrPosition, epsilon, p]

            # 当误差大于某个最大可接受误差时，舍弃该模板
            if epsilon < maxEpsilon:
                acceptedCollection[index] = value
                index += 1

            print(" cal min epsilon and index" + " for " + str(items[0]))
            print("    epsilonindex, harrPosition, epsilon, p = ", value)
            print("    epsilon", epsilon)

        print("one accepted Harr collection done!")
        return acceptedCollection

    def calMinEpsilonIndex(self, harrValueList):

        '''
        在经过排序的harr特征列表上依次扫描，寻找出可以使分类误差最小的位置
        并返回误差的最小值和最小值的位置索引，位置索引即该Harr分类器的分类阈值
        input:
            harrValueList： 经过排序后的harr Value列表
        output:
            epsilon：最小值
            epsilonindex: 最小值的位置

        '''
        epsilonList = []
        minEpsilon = 0.0
        epsilonindex = None
        tPos = 0.0
        tNeg = 0.0
        for i in range(len(harrValueList)):
            if harrValueList[i][1] == 1:
                tPos += 1.0
            else:
                tNeg += 1.0

        tPos = tPos/len(harrValueList)
        tNeg = tNeg/len(harrValueList)

        sPos = 0.0
        sNeg = 0.0

        # 从头到尾依次扫描
        # index 为当前元素
        # 总是比较当前元素之前的正负例的权重

        for index in range(len(harrValueList)):
            for i in range(index):
                value = harrValueList[i][1]
                if value == 1:
                    sPos += 1.0
                else:
                    sNeg += 1.0

            sPos = sPos/len(harrValueList)
            sNeg = sNeg/len(harrValueList)

            epsilonList.append(min((sPos + tNeg - sNeg), (sNeg + tPos - sPos)))
            minEpsilon = min(epsilonList)
            epsilonindex = epsilonList.index(minEpsilon)
            if (sPos + tNeg - sNeg)< (sNeg + tPos - sPos):
                p = 1
            else:
                p = -1

        print("done!")
        return minEpsilon, epsilonindex, p


    def harrCollectionAll(self, edges):
        harrCollection = {}
        selectRegionLength = 24 - edges
        for i in range(selectRegionLength):
            for j in range(selectRegionLength):
                key = str(i * selectRegionLength + j)
                position = (i, j)
                point1 = (i + edges/2, j + edges/2)
                point2 = (i + edges/2, j + edges)
                point3 = (i + edges, j + edges/2)
                point4 = (i + edges, j + edges)
                harrCollection[key] = [position, point1, point2, point3, point4]

        return harrCollection

    def calIntegration(self, startP, endP, img):

        # 给出从左上角 startP 到右下角 endP 的积分图计算公式
        SAT = 0.0
        startY, startX = int(startP[0]), int(startP[1])
        endY, endX = int(endP[0]), int(endP[1])
        for i in range(endY - startY):
            y = startY + i
            for j in range(endX - startX):
                x = startX + j
                SAT += img[y][x]


        return SAT

    def calBlockABCD(self, img, harrPosition):

        '''
            .......................
            .          .          .
            .     A    .    B     .
            .          .          .
            ............ 1 ........2
            .          .          .
            .     C    .    D     .
            .          .          .
            ............ 3 ........4

        '''
        [position, point1, point2, point3, point4] = harrPosition
        SAT1 = self.calIntegration(position, point1, img)
        SAT2 = self.calIntegration(position, point2, img)
        SAT3 = self.calIntegration(position, point3, img)
        SAT4 = self.calIntegration(position, point4, img)
        SATA = SAT1
        SATB = SAT2 - SATA
        SATC = SAT3 - SATA
        SATD = SAT4 - SATA - SATB - SATC


        return SATA, SATB, SATC, SATD

    def HarrAFeatures(self, SATA, SATB, SATC, SATD):
        SATLeft = SATA + SATC
        SATRight = SATB + SATD
        Value =   SATLeft - SATRight

        return Value


    '''
    如果需要使用其他Harr特征可以另外定义
    def HarrBFeatures(self, SATA, SATB, SATC, SATD):
        pass

    def HarrCFeatures(self, SATA, SATB, SATC, SATD):
        pass

    def HarrDFeatures(self, SATA, SATB, SATC, SATD):
        pass
    '''

if __name__ == '__main__':

    imgCollection = readImgCollection(posPath, negPath)
    harrFeatures = HarrFeatures(imgCollection, epsilonList)
    harrFeatures.acceptedHarrCollectionAll()
    print("Done!")
