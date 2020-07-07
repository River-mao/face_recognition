# author: River Mao
# 实现adaboost算法

import cv2
import time
import numpy as np
import numba
from numba import jit
from .Harr import HarrFeatrues
from tkinter import _flatten
from .util import readImgCollection, normalization, load_dir, save_dir, imgShow

class strongClassifier(object):

    def __init__(self, harr, number=20, alphaName=None, alphaPath=None, if_train=True):
        """
        输入： harr, 将定义harr模板的类传入，可调用harr的内置属性
              number, 组成一个强分类器的弱分类器个数，默认20
        """

        self.alphaName = alphaName
        self.alphaPath = alphaPath
        #self.harr = harr
        # 弱分类器的数目
        self.number = number
        self.acceptedHarrY2Features = harr.acceptedHarrY2Features
        self.calIntForOne = harr.calIntForOne
        self.harrY2 = harr.harrY2

        # 选择是否加载已训练参数，或者重新训练
        if not if_train:
            self.alpha = load_dir(alphaPath + '/' +alphaName + '.pkl')
        else:
            self.alpha = []


    def cascadePredict(self, img=None, cascadeNum=None, dsize=(24, 24), step=24, scanEdges=300, creditableNum=20):
        #print("predicting...!")

        # 输入的应该是图片的路径
        # 以灰度图的格式读入图片
        # 并记录图片的大小
        img = cv2.imread(img)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgShape = np.shape(grayimg)

        #扫描窗集合

        scanCollection = {}
        #t1 = time.time()
        for y in range(int(48/step), int((imgShape[0] - 2*scanEdges)/step)):
            for x in range(int(80/step), int((imgShape[1] - scanEdges)/step)):
                scan = cv2.resize(grayimg[step*y: step*y+scanEdges, step*x: step*x+scanEdges], dsize=dsize)
                key = (step *y, step *x)
                scanCollection[key] = scan
        #t2 = time.time()
        #print("读取子块图片耗费时间:{}".format(t2-t1))

        posCollection = scanCollection
        tmpCollection = {}

        for rank in range(cascadeNum):
            t1 = time.time()
            print("cascade:{}, posCollection:{} pieces, scanEdges:{}".format(rank + 1, len(posCollection), scanEdges))

            index = 0
            for startPoint in posCollection.keys():
                index +=1
                print("+---------+ piece: {}/{}".format(index, len(posCollection)))
                #t3 = time.time()
                prediction = self.finalClassifier(img=posCollection[startPoint], rank=rank)
                #t4 = time.time()
                #print("一次预测所需时间:{}".format(t4-t3))
                if prediction > 0 :
                    tmpCollection[startPoint] = posCollection[startPoint]

            posCollection = tmpCollection
            tmpCollection = {}
            t2=time.time()
            print("第{}级预测用时：{}".format(rank, t2-t1))
            if len(posCollection) < creditableNum:
                break

        # 框融合
        y = 0
        x = 0
        for point1 in posCollection.keys():
            y += point1[0]
            x += point1[1]
            y2 = point1[0] + scanEdges
            x2 = point1[1] + scanEdges
            point2 = (y2, x2)
            pt1 = (point1[1], point1[0])
            pt2 = (point2[1], point2[0])
            cv2.rectangle(img=img, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=2)

        # 显示可接所有可接受的框
        imgShow(img=img)

        y = int(y/len(posCollection))
        x = int(x/len(posCollection))

        pt1 = (x, y)
        pt2 = (x+scanEdges, y+scanEdges)
        cv2.rectangle(img=img, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=2)
        imgShow(img=img)
        return  grayimg[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    def finalClassifier(self, img, rank=0):
        #t1 = time.time()
        fx = 0
        for iteration in range(rank*self.number, (rank+1)*self.number):
            #t2 = time.time()
            predict = self.stump(img=img, harrFeature=self.acceptedHarrY2Features[iteration])
            #t3 = time.time()
            #print("Time for stump classifier:{}s".format(t3 - t2))
            fx += self.alpha[iteration] * predict

        Gx = np.sign(fx)
        #t4 = time.time()
        #print("Time for final classifier:{}s".format(t4-t1))
        return Gx

    def boosting(self, rank=None, harrClass='2y', imgCollection=None):
        imgNum = len(imgCollection)
        # 数据集权重矩阵
        # 由于最后还会更新一次权重，需多出一维
        DList = (self.number+1)*[imgNum * [0]]
        # 标签矩阵
        labelList = imgNum*[0]
        # 预测值矩阵
        predictList = imgNum*[0]
        # 分类器权重矩阵
        alphaList = self.number * [0]
        # 误差矩阵
        emList  = self.number * [0]

        for index in range(imgNum):
            DList[0][index] = 1/imgNum

        for iteration in range(self.number):
            print("iteration : {}".format(iteration))

            if harrClass == '2y':
                index = rank*self.number + iteration
                curHarrFeatures = self.acceptedHarrY2Features[index]
                imgIndex = 0

                # 计算特征值
                print("cal eigen value ...")
                for imgName in imgCollection.keys():
                    print("cal eigen value, imgNum:{}/{}".format(imgIndex, imgNum))
                    img = normalization(cv2.imread(imgName, cv2.COLOR_BGR2GRAY))
                    #img = cv2.imread(imgName, cv2.COLOR_BGR2GRAY)
                    # 预测值， 1或-1
                    predictList[imgIndex] = self.stump(img=img, harrFeature=curHarrFeatures)
                    labelList[imgIndex] = imgCollection[imgName]
                    imgIndex +=1

                # 对分类错误的图片的权重进行累加，即得到当前次迭代的em值
                print("cal em ...")
                for imgIndex in range(imgNum):
                    if predictList[imgIndex] != labelList[imgIndex]:
                        emList[iteration] += DList[iteration][imgIndex]

                # 计算alpha
                alphaList[iteration] = 0.5 * np.log((1-emList[iteration])/emList[iteration])
                print("alpha List:{}".format(alphaList))

                # 计算规范化因子
                Zm = 0
                for imgIndex in range(imgNum):
                    Zm += DList[iteration][imgIndex]\
                          * np.exp(-alphaList[iteration]*labelList[imgIndex]*predictList[imgIndex])

                # 更新权重
                print("update weight...")
                for imgIndex in range(imgNum):
                    DList[iteration+1][imgIndex] = (DList[iteration][imgIndex]
                                                    *np.exp(-alphaList[iteration]
                                                    *labelList[imgIndex]
                                                    *predictList[imgIndex]))/Zm


        # (rankNum, stumpNum)
        self.alpha.append(alphaList)
        # 将列表拉成一维
        self.alpha = list(_flatten(self.alpha))
        print("alpha len:{}".format(len(self.alpha)))
        save_dir(self.alpha, self.alphaName, self.alphaPath)
        print("rank {} alpha list saved!".format(rank))

            #harrName = self.harrName + '_rank_' + str(rank)
            #self.ownHarrFeatures = self.harr.acceptedHarrY2Features[rank*self.number:(rank+1)*self.number]
            #save_dir(self.ownHarrFeatures, harrName, self.savePath)


    def stump(self, img, harrClass='2y', harrFeature=None):

        """
        功能：实现harr特征的分类树桩
        输入：img：与 harr 规定的大小一致的单通道灰度图
              harrFeature: harr模板
        """
        # _, 特征值， 右上角顶点， 左下角顶点， 不等号方向
        _, eigenValue0, pt1, pt2, p = harrFeature[0], harrFeature[1], harrFeature[2], harrFeature[3], harrFeature[4]
        intDict = self.calIntForOne(img=img)

        #if harrClass == '2y':
        imgEigenValue = self.harrY2(iDict=intDict, pt1=pt1, pt2=pt2)

        # 待扩展
        # if 2x, 3x, 3y, point...

        # 实现原理见报告
        if p == 1:
            if imgEigenValue > eigenValue0:
                predict = -1
            else:
                predict = 1
        else:
            if imgEigenValue > eigenValue0:
                predict = 1
            else:
                predict = -1

        return predict

if __name__ == '__main__':

    rankNum = 30
    number = 10
    posPath = '../../data/samples/size_24_pt_90_nt_40_num_2/pos/'
    negPath = '../../data/samples/size_24_pt_90_nt_40_num_2/neg/'
    intdictPath = '../../checkpoints/intDir'
    posIntDictName = 'size_24_pt_90_nt_40_num_2_pos'
    negIntDictName = 'size_24_pt_90_nt_40_num_2_neg'
    harrFeaturePath = '../../checkpoints/harrFeatures/size_24_pt_90_nt_40_num_2'
    alphaPath = '../../checkpoints/classifierParam'
    alphaName = 'classifier_alpha'
    testimgPath = '../data/samples/size_24_pt_90_nt_40_num_2/pos/dujiang_18_2p.png'

    # 储存了图片名和标签的字典, 包括了父文件夹posPath, negPath
    imgCollection = readImgCollection(posPath=posPath, negPath=negPath)

    harr=HarrFeatrues(posPath=posPath, negPath=negPath,
                      intdictPath=intdictPath, harrFeaturePath=harrFeaturePath,
                      posIntDictName=posIntDictName, negIntDictName=negIntDictName,
                      if_intDictExist=True, cal_y2=False, if_y2=1)

    harrNum = len(harr.acceptedHarrY2Features)
    #rankNum = int(harrNum/number)
    alphaName = alphaName + '_rank_' + str(rankNum) + '_num_' + str(number)

    # 训练将if_train置为True
    Classifier = strongClassifier(harr=harr, number=number, if_train=True,
                                  alphaPath=alphaPath, alphaName=alphaName)

    #Classifier.boosting(rank=0)
    for rank in range(rankNum):
        print("rank:{}/{}".format(rank, rankNum))
        Classifier.boosting(rank=rank)

    testImg = normalization(cv2.imread(testimgPath, cv2.COLOR_BGR2GRAY))

    predicttion = Classifier.finalClassifier(testImg)
    print("the prediction is:", predicttion)
    print("Done!")