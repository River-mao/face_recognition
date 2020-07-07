# Author: River Mao

# 功能： 使用积分图 实现Harr特征，并返回Harr特征值
# 一个Harr模板是：“待检测区域大小”、 “模板位置” 和“模板大小”三个参量的函数
# 对于这三个参数我们有以下规定：
# 1 待检测区域:
#   一般由上游任务给定， 在该任务中我们规定为（24， 24）
# 2 模板大小与模板位置
#   使用枚举的方法， 列出所有的harr特征
#   模板的位置由左上角顶点确定
#   模板的大小由边长确定（在该任务中我们设置模板最小宽度为）
import os
import cv2
import numpy as np
from. util import save_dir, load_dir, normalization, extractFilename

# x2
maxEpsilon1 = 0.4
# x3
maxEpsilon2 = 0.4
# y2
maxEpsilon3 = 0.4
# y3
maxEpsilon4 = 0.4
# point
maxEpsilon5 = 0.4

posPath = '../../data/samples/size_24_pt_90_nt_40_num_2/pos'
negPath = '../../data/samples/size_24_pt_90_nt_40_num_2/neg'
intdictPath = '../../checkpoints/intDir'
posIntDictName = 'size_24_pt_90_nt_40_num_2_pos'
negIntDictName = 'size_24_pt_90_nt_40_num_2_neg'
harrFeaturePath = '../../checkpoints/harrFeatures/size_24_pt_90_nt_40_num_2'

class HarrFeatrues(object):

    def __init__(self, posPath=None, negPath=None, intdictPath=None, harrFeaturePath=None,
                 if_x2=0, if_x3=0, if_y2=0, if_y3=0, if_point=0, if_intDictExist=False,
                 minEdge=6, cal_x2=False, cal_y2=True, cal_x3=False, cal_y3=False, cal_point=False,
                 posIntDictName=None, negIntDictName=None, dsize = (24, 24), featureNum=1000):
        """
        功能：
            1 积分图运算，并保存结果，以便在后续的 harr 特征值的计算中直接调用
            2 计算harr2x, harr2y, harr3x, harr3y, harr point五种特征
            3 根据可接受误差筛选一定数量的harr模板， 默认值为1000，并储存，
              每个模板的储存格式为：（最小误差, 特征值（分类树桩）, 模板右上角顶点, 模板左下角顶点, 不等号方向）
        输入：
            路径： posPath：正例集路径
                  negPath：负例集路径
                  intdictPath：保存积分图的文件夹
                  harrFeaturePath：保存harr特征值的文件夹
                  posIntDictName:正例积分图的文件名
                  negIntDictName：负例积分图文件名

            图片（模板）大小： dsize = (24, 24)，可直接修改，但是需保证为正方形
                              minEdge， 最小模板边长，理论最小边长为3， 可增大该值减少迭代次数

            标记：
                if_intDictExist：是否存在图片的积分图数据， 若不存在则重新计算，
                                 即设置为False时，初始时会重新计算一次图片的像素积分，并保存
                                 设置为True时，需给入文件路径，加载数据

                if_x2=0, if_x3=0, if_y2=0, if_y3=0, if_point=0：用于标记已有的harr特征

                cal_x2, cal_y2, cal_x3, cal_y3, cal_point： 用于标记计算哪种harr特征，由于PC内存的限制，一次只设置一个为True

        """

        # 图片归一化大小
        self.dsize = dsize

        # 正负图片集的路径
        self.posPath = posPath
        self.negPath = negPath

        # 正负图片集图片名
        self.posNameList = os.listdir(self.posPath)
        self.negNameList = os.listdir(self.negPath)

        # 正负图片集积分图数据的路径
        posIntdictPath = intdictPath + '/' + posIntDictName + '.pkl'
        negIntdictPath = intdictPath + '/' + negIntDictName + '.pkl'

        # 已有图片集的积分数据时，从目标路径加载数据
        if if_intDictExist:
            print("有可用积分图数据，加载数据")
            self.intDictPos = load_dir(posIntdictPath)
            self.intDictNeg = load_dir(negIntdictPath)
            print("积分图数据加载完成！")
        # 无积分数据时，计算积分数据并保存于目标路径
        else:
            print("无可用积分图数据，计算积分图")
            self.intDictPos, self.intDictNeg = self.calIntForAll()
            save_dir(dir=self.intDictPos, name=posIntDictName, DirPath=intdictPath)
            save_dir(dir=self.intDictNeg, name=negIntDictName, DirPath=intdictPath)
            print("积分图计算完成！")

        # 用于保存或者读取harr特征的文件名
        self.x2Name = 'harrx2'
        self.y2Name = 'harry2'
        self.x3Name = 'harrx3'
        self.y3Name = 'harry3'
        self.pointName = 'harrpoint'

        # 满足条件的harr模板集合
        self.acceptedHarrX2Features = []
        self.acceptedHarrX3Features = []
        self.acceptedHarrY2Features = []
        self.acceptedHarrY3Features = []
        self.acceptedHarrPointFeatures = []

        # 已有模板数据时， 载入数据
        if if_x2:
            print("有可用harrx2特征数据")
            self.acceptedHarrX2Features = load_dir(harrFeaturePath+'/'+self.x2Name +'.pkl')
        if if_x3:
            print("有可用harrx3特征数据")
            self.acceptedHarrX3Features = load_dir(harrFeaturePath+'/'+self.x3Name +'.pkl')
        if if_y2:
            print("有可用harry2特征数据")
            self.acceptedHarrY2Features = load_dir(harrFeaturePath+'/'+self.y2Name +'.pkl')
        if if_y3:
            print("有可用harry3特征数据")
            self.acceptedHarrY3Features = load_dir(harrFeaturePath+'/'+self.y3Name +'.pkl')
        if if_point:
            print("有可用harrpoint特征数据")
            self.acceptedHarrPointFeatures = load_dir(harrFeaturePath+'/'+self.pointName+'.pkl')

        # 无数据时， 重新计算数据，并保存
        if  (if_x2+if_x3+if_y2+if_y3+if_point)==0:
            print("无可用harr特征，计算特征")
            self.calHarrFeatures(minEdge=minEdge, cal_x2=cal_x2, cal_y2=cal_y2,
                                 cal_x3=cal_x3, cal_y3=cal_y3, cal_point=cal_point,featuresNum=featureNum)
            print("特征计算完成, 并已完成筛选保存！")


    def calHarrFeatures(self, minEdge=3, cal_x2=False, cal_y2=True, cal_x3=False, cal_y3=False, cal_point=False, featuresNum=1000):

        # 由于内存限制， 通过设置cal_x2=False, cal_y2=True, cal_x3=False, cal_y3=False, cal_point=False限定每次只算一个特征
        # 窗口大小， 即图片大小
        windowSize = int(self.dsize[0])
        # 储存可使用的模板参数

        for yStart in range(windowSize - minEdge):
            for xStart in range(windowSize - minEdge):
                print("Start point:({},{})/({},{})".
                      format(yStart, xStart, windowSize - minEdge, windowSize - minEdge))

                pt1 = (yStart, xStart)
                for yEnd in range((minEdge + yStart), windowSize):
                    for xEnd in range(minEdge + xStart, windowSize):
                        # 以harr特征的左上角以及右下角顶点为参数为文件名保存数据
                        pt2 = (yEnd, xEnd)
                        Y = yEnd - yStart
                        X = xEnd - xStart
                        # 中间变量， 用于暂时储存各harr特征的特征值
                        # eigen value - label pair list
                        elPaitListX2=[]
                        elPaitListX3=[]
                        elPaitListY2=[]
                        elPaitListY3=[]
                        elPaitListPoint=[]

                        # 遍历正样本集， 计算每一个正样本在当前Harr特征上的特征值
                        for posName in self.posNameList:
                            curDir = self.intDictPos[posName]
                            # 如果X边可以被2整除, 可计算x2特征
                            # 由于x2特征对人脸区分能力较弱， 因此省略不算
                            if cal_x2:
                                if X % 2 == 0:
                                    curEigenvaluesX2 = self.harrX2(iDict=curDir, pt1=pt1, pt2=pt2)
                                    elPaitListX2.append((curEigenvaluesX2, 1))

                            # 如果Y边可以被2整除, 可计算y2特征,
                            if cal_y2:
                                if Y % 2 == 0:
                                    curEigenvaluesY2 = self.harrY2(iDict=curDir, pt1=pt1, pt2=pt2)
                                    elPaitListY2.append((curEigenvaluesY2, 1))

                            # 如果X边可以被3整除, 可计算x3特征
                            if cal_x3:
                                if X % 3 == 0:
                                   curEigenvaluesX3 = self.harrX3(iDict=curDir, pt1=pt1, pt2=pt2)
                                   elPaitListX3.append((curEigenvaluesX3, 1))

                            # 如果Y边可以被3整除, 可计算y3特征
                            if cal_y3:
                                if Y % 3 == 0:
                                    curEigenvaluesY3 = self.harrY3(iDict=curDir, pt1=pt1, pt2=pt2)
                                    elPaitListY3.append((curEigenvaluesY3, 1))

                            # 如果X, Y边都可以被3整除, 可计算point特征
                            if cal_point:
                                if (X % 3 == 0) & (Y % 3 == 0):
                                    curEigenvaluesPoint = self.harrPoint(iDict=curDir, pt1=pt1, pt2=pt2)
                                    elPaitListPoint.append((curEigenvaluesPoint, 1))

                        # 遍历负样本集， 计算每一个负样本在当前Harr特征上的特征值
                        for negName in self.negNameList:
                            curDir = self.intDictNeg[negName]
                            # 如果X边可以被2整除, 可计算x2特征
                            if cal_x2:
                                if X % 2 == 0:
                                    curEigenvaluesX2 = self.harrX2(iDict=curDir, pt1=pt1, pt2=pt2)
                                    elPaitListX2.append((curEigenvaluesX2, -1))

                            # 如果Y边可以被2整除, 可计算y2特征
                            if cal_y2:
                                if Y % 2 == 0:
                                    curEigenvaluesY2= self.harrY2(iDict=curDir, pt1=pt1, pt2=pt2)
                                    elPaitListY2.append((curEigenvaluesY2, -1))

                            # 如果X边可以被3整除, 可计算x3特征
                            if cal_x3:
                                if X % 3 == 0:
                                   curEigenvaluesX3 = self.harrX3(iDict=curDir, pt1=pt1, pt2=pt2)
                                   elPaitListX3.append((curEigenvaluesX3, -1))

                            # 如果Y边可以被3整除, 可计算y3特征
                            if cal_y3:
                                if Y % 3 == 0:
                                    curEigenvaluesY3 = self.harrY3(iDict=curDir, pt1=pt1, pt2=pt2)
                                    elPaitListY3.append((curEigenvaluesY3, -1))

                            # 如果X, Y边都可以被3整除, 可计算point特征
                            if cal_point:
                                if (X % 3 == 0) & (Y % 3 == 0):
                                    curEigenvaluesPoint = self.harrPoint(iDict=curDir, pt1=pt1, pt2=pt2)
                                    elPaitListPoint.append((curEigenvaluesPoint, -1))

                        if len(elPaitListX2) !=0:
                            # 对特征值进行排序
                            # 并对模板进行筛选， 只有当分类错误率小于某个可接受误差时，才加模板加入到可用模板中
                            elPaitListX2 = sorted(elPaitListX2, reverse=True)
                            minEpsilon, epsilonindex, p = self.calMinEpsilonIndex(elPaitListX2)
                            print("x2, pt2:{}, min epsilon {}".format(pt2, minEpsilon))

                            if minEpsilon < maxEpsilon1:
                                self.acceptedHarrX2Features.append((minEpsilon, elPaitListX2[epsilonindex][0], pt1, pt2, p))
                                print("acceptedHarrX2Features len:{}".format(len(self.acceptedHarrX2Features)))

                                if len(self.acceptedHarrX2Features)>featuresNum:
                                    self.acceptedHarrX2Features = sorted(self.acceptedHarrX2Features)
                                    save_dir(self.acceptedHarrX2Features, self.x2Name, harrFeaturePath)

                                    return

                        if len(elPaitListX3) !=0:
                            elPaitListX3 = sorted(elPaitListX3, reverse=True)
                            minEpsilon, epsilonindex, p = self.calMinEpsilonIndex(elPaitListX3)
                            print("y2, pt2:{}, min epsilon {}, index:{}, p:{}".format(pt2, minEpsilon, epsilonindex, p))
                            if minEpsilon < maxEpsilon2:
                                self.acceptedHarrX3Features.append((minEpsilon, elPaitListX3[epsilonindex][0], pt1, pt2, p))
                                print("acceptedHarrX3Features len:{}".format(len(self.acceptedHarrX3Features)))

                                if len(self.acceptedHarrX3Features)>featuresNum:
                                    self.acceptedHarrY2Features = sorted(self.acceptedHarrY2Features)
                                    save_dir(self.acceptedHarrX3Features, self.x3Name, harrFeaturePath)

                                    return

                        if len(elPaitListY2) != 0:
                            elPaitListY2 = sorted(elPaitListY2, reverse=True)
                            minEpsilon, epsilonindex, p = self.calMinEpsilonIndex(elPaitListY2)
                            print("x3, pt2:{}, min epsilon {}, index:{}/{}, eigenvalue:{},  p:{}"
                                  .format(pt2, minEpsilon, epsilonindex, len(elPaitListY2), elPaitListY2[epsilonindex][0], p))
                            if minEpsilon < maxEpsilon3:
                                self.acceptedHarrY2Features.append((minEpsilon, elPaitListY2[epsilonindex][0], pt1, pt2, p))
                                print("acceptedHarrY2Features len:{}".format(len(self.acceptedHarrY2Features)))

                                # 当模板数目大于2000时，停止计算
                                if len(self.acceptedHarrY2Features)>featuresNum:
                                    self.acceptedHarrX3Features = sorted(self.acceptedHarrX3Features)
                                    save_dir(self.acceptedHarrY2Features, self.y2Name, harrFeaturePath)

                                    return

                        if len(elPaitListY3) != 0:
                            elPaitListY3 = sorted(elPaitListY3, reverse=True)
                            minEpsilon, epsilonindex, p = self.calMinEpsilonIndex(elPaitListY3)
                            print("y3, pt2:{}, min epsilon {}".format(pt2, minEpsilon))
                            if minEpsilon < maxEpsilon4:
                                self.acceptedHarrY3Features.append((minEpsilon, elPaitListY3[epsilonindex][0], pt1, pt2, p))
                                print("acceptedHarrY3Features len:{}".format(len(self.acceptedHarrY3Features)))

                                if len(self.acceptedHarrY3Features)>featuresNum:
                                    self.acceptedHarrY3Features = sorted(self.acceptedHarrY3Features)
                                    save_dir(self.acceptedHarrY3Features, self.y3Name, harrFeaturePath)

                                    return

                        if len(elPaitListPoint) != 0:
                            elPaitListPoint = sorted(elPaitListPoint, reverse=True)
                            minEpsilon, epsilonindex, p = self.calMinEpsilonIndex(elPaitListPoint)
                            print("point, pt2:{}, min epsilon {}".format(pt2, minEpsilon))
                            if minEpsilon < maxEpsilon5:
                                self.acceptedHarrPointFeatures.append((minEpsilon, elPaitListPoint[epsilonindex][0], pt1, pt2, p))
                                print("acceptedHarrPointFeatures len:{}".format(len(self.acceptedHarrPointFeatures)))

                                if len(self.acceptedHarrPointFeatures)>featuresNum:
                                    self.acceptedHarrPointFeatures = sorted(self.acceptedHarrPointFeatures)
                                    save_dir(self.acceptedHarrPointFeatures, self.pointName, harrFeaturePath)

                                    return



    def calMinEpsilonIndex(self, List=None):
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
        # 所有正例的均值
        tPos = 0.0
        # 所有负例的均值
        tNeg = 0.0
        for i in range(len(List)):
            if List[i][1] == 1:
                tPos += 1.0
            else:
                tNeg += 1.0

        tPos = tPos/len(List)
        tNeg = tNeg/len(List)

        p=1

        # 从头到尾依次扫描
        # index 为当前元素
        for index in range(len(List)):
            sPos = 0.0
            sNeg = 0.0
            for i in range(index):
                if List[i][1] == 1:
                    sPos += 1.0
                else:
                    sNeg += 1.0

            sPos = sPos/len(List)
            sNeg = sNeg/len(List)

            epsilonList.append(min((sPos + tNeg - sNeg), (sNeg + tPos - sPos)))
            minEpsilon = min(epsilonList)
            epsilonindex = epsilonList.index(minEpsilon)
            if (sPos + tNeg - sNeg) < (sNeg + tPos - sPos):
                p = 1
            else:
                p = -1

        return minEpsilon, epsilonindex, p

    def calIntForAll(self):
        """
        功能： 针对图片集计算积分和

        """
        intDictPos = {}
        intDictNeg = {}
        posLength = len(self.posNameList)
        negLength = len(self.negNameList)
        print("{}张正例图片待计算， {}张负例图片待计算".format(posLength, negLength))
        posIndex = 0
        negIndex = 0
        for posName in self.posNameList:
            posImgPath = self.posPath + '/' + posName
            grayImg = cv2.cvtColor(cv2.imread(posImgPath), cv2.COLOR_BGR2GRAY)
            posImg = normalization(cv2.resize(grayImg, self.dsize))
            intDictPos[posName] = self.calIntForOne(img=posImg)
            posIndex +=1
            print("{}/{}张正例积分图计算完成".format(posIndex, posLength))

        for negName in self.negNameList:
            negImgPath = self.negPath + '/' + negName
            grayImg = cv2.cvtColor(cv2.imread(negImgPath), cv2.COLOR_BGR2GRAY)
            negImg = normalization(cv2.resize(grayImg, self.dsize))
            intDictNeg[negName] = self.calIntForOne(img=negImg)
            negIndex +=1
            print("{}/{}张负例积分图计算完成".format(negIndex, negLength))

        return intDictPos, intDictNeg

    def calIntForOne(self, img=None):
        """
        功能：针对单张图片，实现从（0，0）-->(endY, endX)的像素和，并以字典的形式返回
             键为坐标点，值为积分和，即{（y, x）: Integration Value}
             ................
             .      .       .
             .      .       .
             ........(y, x) .  edge
             .              .
             ................
                    edge
        输入：
            img: 输入图片， 一般默认为是（24，24）
        输出：
            intDictOne :积分和 字典
        """
        intDictOne = {}
        edge = np.shape(img)[0]
        for endY in range(edge):
            for endX in range(edge):
                key = (endY, endX)
                intValue = 0
                for y in range(endY):
                    for x in range(endX):
                        intValue += img[y][x]
                intDictOne[key] = intValue

        return intDictOne

    def harrX2(self, iDict=None, pt1=None, pt2=None):
        """
        功能： harr X2 特征值的计算
        输入：
            iDict：integrationDict,储存图片积分和的字典
            pt1: 模板左上角顶点（y1, x1）
            pt2: 模板右下角顶点（y2, x2）
            并且假设只在x2-x1为偶数时计算harr A模板
        输出：
            Eigenvalues： 该特征的特征值
        """
        #  harr X2                                 在图片中的一般情况如下：
        # +---------------------+              +-------------------------+
        # + pt1    pt3    pt4   +              +  +-----+-----+-----+    +
        # +   +-----+-----+     +              +  +-(1)-+-(2)-+-(3)-+    +
        # +   +-(w)-+-(b)-+     +              +  +-----+1----+3----+4   +
        # +   +-----+-----+     +              +  +-(4)-+-(w)-+-(b)-+    +
        # +  pt5   pt6    pt2   +              +  +-----+5----+6----+2   +
        # +---------------------+              +-------------------------+
        # Eigenvalues=pix_sum(w)-pix_sum(b)
        # integration(x) = I(x), 从(0,0)到右下顶点(y, x)的积分
        # pix_sum(w) = I(6)-I(5)-I(3)+I(1)
        # pix_sum(b) = I(2)-I(6)-I(4)+I(3)
        y1, x1 = int(pt1[0]), int(pt1[1])
        y2, x2 = int(pt2[0]), int(pt2[1])
        pt3 = (y1, 0.5*(x2-x1))
        pt4 = (y1, x2)
        pt5 = (y2, x1)
        pt6 = (y2, 0.5*(x2-x1))
        pixSumW = iDict[pt6]-iDict[pt5]-iDict[pt3]+iDict[pt1]
        pixSumB = iDict[pt2]-iDict[pt6]-iDict[pt4]+iDict[pt3]
        Eigenvalues = pixSumW - pixSumB

        return Eigenvalues

    def harrY2(self, iDict=None, pt1=None, pt2=None):
        """
        功能： harr Y2 特征值的计算
        输入：
            iDict：integrationDict,储存图片积分和的字典
            pt1: 模板左上角顶点（y1, x1）
            pt2: 模板右下角顶点（y2, x2）
            并且假设只在y2-y1为偶数时计算harr B模板
        输出：
            Eigenvalues： 该特征的特征值
        """
        #  harr Y2                                  在图片中的一般情况如下：
        # +---------------------+                +-------------------------+
        # + pt1      pt3        +                +  +-----+-----+1----+3   +
        # +   +-----+           +                +  +-(1)-+-(2)-+-(b)-+    +
        # +   +-(b)-+           +                +  +-----+-----+4----+5   +
        # +pt4+-----+pt5        +                +  +-(3)-+-(4)-+-(w)-+    +
        # +   +-(w)-+           +                +  +-----+-----+6----+2   +
        # +pt6+-----+pt2        +                +                         +
        # +---------------------+                +-------------------------+
        # Eigenvalues=pix_sum(w)-pix_sum(b)
        # integration(x) = I(x), 从(0,0)到右下顶点(y, x)的积分
        # pix_sum(w) = I(2)-I(6)-I(5)+I(4)
        # pix_sum(b) = I(5)-I(4)-I(3)+I(1)
        # y1, x1 = int(pt1[0]), int(pt1[1])
        # y2, x2 = int(pt2[0]), int(pt2[1])
        pt3 = (pt1[0], pt2[1])
        pt4 = (0.5*(pt2[0]-pt1[0]), pt1[1])
        pt5 = (0.5*(pt2[0]-pt1[0]), pt2[1])
        pt6 = (pt2[0], pt1[1])
        # pixSumW = iDict[pt2]-iDict[pt6]-iDict[pt5]+iDict[pt4]
        # pixSumB = iDict[pt5]-iDict[pt4]-iDict[pt3]+iDict[pt1]
        # Eigenvalues = pixSumW - pixSumB
        Eigenvalues = iDict[pt2]-iDict[pt6]-iDict[pt5]+iDict[pt4]-(iDict[pt5]-iDict[pt4]-iDict[pt3]+iDict[pt1])

        return Eigenvalues

    def harrX3(self, iDict=None, pt1=None, pt2=None):
        """
        功能： HarrX3特征的计算, 当x2-x1可以被3整除的时候才计算HarrX3
        """
        # harr X3
        # +-------------------+
        # + pt1 pt3 pt4 pt5   +
        # +  +---+---+---+    +
        # +  +-w1+-b-+-w2+    +
        # +  +---+---+---+    +
        # + pt6 pt7 pt8 pt2   +
        # +-------------------+
        #  Eigenvalues=pix_sum(w)-2*pix_sum(b)
        y1, x1 = int(pt1[0]), int(pt1[1])
        y2, x2 = int(pt2[0]), int(pt2[1])
        diffX = (x2-x1)/3
        pt3 = (y1, x1+diffX)
        pt4 = (y1, x1+2*diffX)
        pt5 = (y1, x2)
        pt6 = (y2, x1)
        pt7 = (y2, x1+diffX)
        pt8 = (y2, x1+2*diffX)
        pixSumW1 = iDict[pt7]-iDict[pt6]-iDict[pt3]+iDict[pt1]
        pixSumW2 = iDict[pt2]-iDict[pt8]-iDict[pt5]+iDict[pt4]
        pixSumB  = iDict[pt8]-iDict[pt7]-iDict[pt4]+iDict[pt3]
        Eigenvalues = pixSumW1+pixSumW2-2*pixSumB

        return Eigenvalues

    def harrY3(self, iDict=None, pt1=None, pt2=None):
        """
        功能： HarrY3特征的计算, 当y2-y1可以被3整除的时候才计算HarrY3
        """
        # +-------------+
        # +p1+-----+p3  +
        # +  +--w1-+    +
        # +p4+-----+p5  +
        # +  +--b--+    +
        # +p6+-----+p7  +
        # +  +--w2-+    +
        # +p8+-----+p2  +
        # +-------------+
        y1, x1 = int(pt1[0]), int(pt1[1])
        y2, x2 = int(pt2[0]), int(pt2[1])
        diffY = (y2-y1)/3
        pt3 = (y1, x2)
        pt4 = (y1+diffY, x1)
        pt5 = (y1+diffY, x2)
        pt6 = (y1+2*diffY, x1)
        pt7 = (y1+2*diffY, x2)
        pt8 = (y2, x1)
        pixSumW1 = iDict[pt5]-iDict[pt4]-iDict[pt3]+iDict[pt1]
        pixSumW2 = iDict[pt2]-iDict[pt8]-iDict[pt7]+iDict[pt6]
        pixSumB  = iDict[pt7]-iDict[pt6]-iDict[pt5]+iDict[pt4]
        Eigenvalues = pixSumW1+pixSumW2-2*pixSumB

        return Eigenvalues

    def harrPoint(self, iDict=None, pt1=None, pt2=None):
        """
        功能： HarrPoint特征的计算, 当x2-x1, y2-y1可以被3整除的时候才计算HarrPoint特征的计算
        """
        #  p1+-----+-----+-----+ p3
        #    + w1  + w2  + w3  +
        #    +---p4+---p5+-----+
        #    + w4  +  b  + w5  +
        #    +---p6+---p7+-----+
        #    + w6  +  w7 +  w8 +
        #  p8+-----+-----+-----+p2
        y1, x1 = int(pt1[0]), int(pt1[1])
        y2, x2 = int(pt2[0]), int(pt2[1])
        diffY = (y2-y1)/3
        diffX = (x2-x1)/3
        pt3 = (y1, x2)
        pt4 = (y1+diffY, x1+diffX)
        pt5 = (y1+diffY, x1+2*diffX)
        pt6 = (y1+2*diffY, x1+diffX)
        pt7 = (y1+2*diffY, x1+2*diffX)
        pt8 = (y1+3*diffY, x1)
        pixSumB  = iDict[pt7]-iDict[pt5]-iDict[pt6]+iDict[pt4]
        pixSumTotal = iDict[pt2]-iDict[pt3]-iDict[pt8]+iDict[pt1]
        pixSumW = pixSumTotal - pixSumB
        Eigenvalues = pixSumW-9*pixSumB
        return Eigenvalues

if __name__ == '__main__':
    harrFeatures = HarrFeatrues(posPath=posPath, negPath=negPath, intdictPath=intdictPath,harrFeaturePath=harrFeaturePath,
                                posIntDictName=posIntDictName, negIntDictName=negIntDictName,
                                if_intDictExist=True)

    print("done!")



