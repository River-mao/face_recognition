import cv2
import numpy as np
from tool import load_dir, readImgCollection, normalization
from HarrFeatures import HarrFeatures

maxIters = 20

posPath = '../samples/size_24_t_50_num_2/pos/'
negPath = '../samples/size_24_t_50_num_2/neg/'
harr16Path =  '../harrFeature/harrColl16.pkl'
#harr8Path  =  '../harrFeature/harrColl8.pkl'
#harr10Path =  '../harrFeature/harrColl10.pkl'
#harr12Path =  '../harrFeature/harrColl12.pkl'
#harr14Path =  '../harrFeature/harrColl14.pkl'

class strongClassifier(object):

    def __init__(self, harrTemp, imgCollection, maxIters = maxIters, harrFeatures = None):
        '''
           O.......................
            .          .          .
            .     A    .    B     .
            .          .          .
            ............ 1 ........2
            .          .          .
            .     C    .    D     .
            .          .          .
            ............ 3 ........4

        '''
        # harrTemp = [阈值， 模板参数[O, 1, 2, 3, 4]（如上图所示）， 分类误差]
        self.harrTemp = harrTemp
        self.maxIters = maxIters
        self.harrFeatures = harrFeatures
        self.imgCollection = imgCollection
        self.alpha = None
        self.UsedHarrTemp = None

    # Boosting过程
    # 返回Adaboost的alpha值和对应的弱分类器的参数

   def Boosting(self):
        imgNum = len(self.imgCollection)
        # 数据集权重矩阵
        # 由于最后还会更新一次权重，需多出一维
        D = (self.maxIters+1)*[imgNum * [0]]
        # 标签矩阵
        label = imgNum*[0]
        # 预测值矩阵
        predict = imgNum*[0]
        # 分类器权重矩阵
        alpha = self.maxIters * [0]
        # 误差矩阵
        em  = self.maxIters * [0]
        # 记录使用的模板参数
        UsedHarrTemp = []

        for index in range(imgNum):
            D[0][index]= 1/imgNum

        for iteration in range(self.maxIters):
            harrValueCollection = {}
            harrValueList = []
            curHarrTemp = self.harrTemp[iteration]
            UsedHarrTemp.append(curHarrTemp)

            imgindex = 0
            for imgName in self.imgCollection.keys():
                img = normalization(cv2.imread(imgName, cv2.COLOR_BGR2GRAY))
                harrValue, prediction = self.weakClassifier(img, curHarrTemp)
                # harrValueCollection[imgName] = harrValue
                # 记录当前图片在当前模板上的预测值（-1，1）
                predict[imgindex] = prediction
                # 读取当前图片在当前模板上实际的标签（-1，1）
                label[imgindex] = self.imgCollection[imgName]
                imgindex += 1
                # [(value0, label0, imgName0),(value1, label1, imgName1), ...]
                # harrValueList.append((harrValue, self.imgCollection[imgName], imgName))

            # 通过harrValue的值从大到小对所有图片进行排序
            # 由于内部是由元组组成，对应的label及文件名同步排序
            #harrValueList = sorted(harrValueList, reverse=True)
            # 将标签与预测值进行比较
            # 并计算误差值

            for imgindex in range(imgNum):
                if predict[imgindex] != label[imgindex]:
                    em[iteration] += D[iteration][imgindex]

            # 计算alpha
            alpha[iteration] = 0.5* np.log((1-em[iteration])/em[iteration])

            # 计算规范化因子
            Zm = 0
            for imgindex in range(imgNum):
                Zm += D[iteration][imgindex]*np.exp(-alpha[iteration] * label[imgindex] * predict[imgindex])

            # 更新权重
            for imgindex in range(imgNum):
                D[iteration+1][imgindex] = (D[iteration][imgindex]
                                            *np.exp(-alpha[iteration]
                                            *label[imgindex]
                                            *predict[imgindex]))/Zm
        self.alpha = alpha
        self.UsedHarrTemp = UsedHarrTemp


    # 弱分类器，任意一个Harr模板
    def weakClassifier(self, img, curHarrTemp):
        boundHarrValue, harrPosition, _, p = curHarrTemp[0], curHarrTemp[1], curHarrTemp[2], curHarrTemp[3]
        SATA, SATB, SATC, SATD = self.harrFeatures.calBlockABCD(img, harrPosition)
        harrValue = self.harrFeatures.HarrAFeatures(SATA, SATB, SATC, SATD)
        if p == 1:
            if harrValue > boundHarrValue:
                predict = -1
            else:
                predict = 1
        else:
            if harrValue > boundHarrValue:
                predict = 1
            else:
                predict = -1

        return  harrValue, predict

    # 使用Adaboost后组成的强分类器
    def finalClassifier(self, img):
        fx = 0
        for iter in range(self.maxIters):
            _, predict = self.weakClassifier(img, self.UsedHarrTemp[iter])
            fx += self.alpha[iter] * predict

        Gx = np.sign(fx)

        return  Gx

if __name__ == '__main__':

    harr16Dir = load_dir(harr16Path)
    imgCollection = readImgCollection(posPath, negPath)
    harrFeatures = HarrFeatures()
    Classifier = strongClassifier(harr16Dir, imgCollection, maxIters, harrFeatures)
    Classifier.Boosting()
    testImg = normalization(cv2.imread('../samples/testImg/curry1_18.png', cv2.COLOR_BGR2GRAY))
    predicttion = Classifier.finalClassifier(testImg)
    print("Done!")

