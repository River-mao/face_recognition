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

class HarrAFeatures(object):

    def __init__(self):
        # collection的数据结构为字典
        # 每一个模板在collection以左上角的坐标作为值储存
        # 键为模板的序号， 为方便调用
        self.harrCollection18All = self.harrCollectionAll(18)
        self.harrCollection16All = self.harrCollectionAll(16)
        self.harrCollection14All = self.harrCollectionAll(14)
        self.harrCollection12All = self.harrCollectionAll(12)
        self.harrCollection10All = self.harrCollectionAll(10)

        # 满足条件的Harr分类器
        self.harrCollection18 = None
        self.harrCollection16 = None
        self.harrCollection14 = None
        self.harrCollection12 = None
        self.harrCollection10 = None

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
                SAT += img(y, x)

        return SAT

    def calBlockABCD(self, img, harrPosition):
        '''
            .......................
            .          .          .
            .     A    .    B     .
            .          .          .
            .............1.........2
            .          .          .
            .     C    .    D     .
            .          .          .
            .............3.........4

        '''
        [position, point1, point2, point3, point4] = harrPosition
        SAT1 = self.calIntegration(position, point1)
        SAT2 = self.calIntegration(position, point2)
        SAT3 = self.calIntegration(position, point3)
        SAT4 = self.calIntegration(position, point4)
        SATA = SAT1
        SATB = SAT2 - SATA
        SATC = SAT3 - SATA
        SATD = SAT4 - SATA - SATB - SATC

        return SATA, SATB, SATC, SATD

    def HarrAFeatures(self, SATA, SATB, SATC, SATD):
        SATLeft = SATA + SATC
        SATRight = SATB + SATD
        Value = SATLeft - SATRight

        return Value
    