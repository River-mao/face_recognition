import cv2
import numpy as np
from Adaboost import strongClassifier
from HarrFeatures import HarrFeatures
from tool import load_dir, save_dir, readImgCollection, normalization, imgShow, select

posPath = '../samples/size_24_t_30_90_num_3/pos/'
negPath = '../samples/size_24_t_30_90_num_3/neg/'
harr6Path =  '../harrFeature/size_24_t_30_90_num_3/harrColl6.pkl'
harr8Path  =  '../harrFeature/size_24_t_30_90_num_3/harrColl8.pkl'
harr10Path =  '../harrFeature/size_24_t_30_90_num_3/harrColl10.pkl'
harr12Path =  '../harrFeature/size_24_t_30_90_num_3/harrColl12.pkl'
harr14Path =  '../harrFeature/size_24_t_30_90_num_3/harrColl14.pkl'
harr16Path =  '../harrFeature/size_24_t_30_90_num_3/harrColl16.pkl'

class CascadeClassifier(object):
    def __init__(self, StrongClassifier, cascadeNum, imgCollection, harrTempList, HarrFeatures, maxiters, trained=False):
        '''
        input:
            StrongClassifier: 强分类器的类
            HarrFeatures: 构造弱分类器需要的Harr模板的类
            cascadeNum: 级联数目
            imgCollection: 图片名与标签，用于读取图片和标签
            harrTempList: 与cascadeNum对应
        '''
        self.trained = trained
        self.HarrFeatures = HarrFeatures
        self.imgCollection = imgCollection
        self.StrongClassifier = StrongClassifier
        self.cascadeNum = cascadeNum
        self.ClassifierList = []
        self.harrTempList = harrTempList
        self.maxiters = maxiters

    def cascade(self):
        print("cascade!")
        for i in range(self.cascadeNum):
            self.ClassifierList.append("Classifier" + str(i))

        for i in range(self.cascadeNum):
            self.ClassifierList[i] = self.StrongClassifier(self.harrTempList[i],
                                                           self.imgCollection,
                                                           self.maxiters,
                                                           self.HarrFeatures)
        if not self.trained:
            for i in range(self.cascadeNum):
                self.ClassifierList[i].Boosting()
                save_dir(self.ClassifierList[i].alpha, "Classifier" + str(i) + "_alpha", "../varForEachClassifier/step_8_edges_300")
                save_dir(self.ClassifierList[i].UsedHarrTemp, "Classifier" + str(i) + "_temp", "../varForEachClassifier/step_8_edges_300")
                print("Classifier {} init done！".format(i+1))
        else:
            for i in range(self.cascadeNum):
                self.ClassifierList[i].alpha = load_dir("../varForEachClassifier/step_8_edges_300/" + "Classifier" + str(i) + "_alpha" + ".pkl")
                self.ClassifierList[i].UsedHarrTemp = load_dir("../varForEachClassifier/step_8_edges_300/" + "Classifier" + str(i) + "_temp" + ".pkl")
                print("Classifier {} init done！".format(i + 1))

        print("cascade done!")


    def predict(self,inputImg , step = 4, scanEdges = 240, creditableNum = 15):
        print("predict!")
        # 输入的应该是图片的路径
        # 以灰度图的格式读入图片
        # 并记录图片的大小

        inputImg = cv2.imread(inputImg)
        img = cv2.cvtColor(inputImg, cv2.COLOR_RGB2GRAY)
        imgShape = np.shape(img)
        print("Img size:", imgShape)

        scanCollection = {}

        for y in range(int((imgShape[0] - scanEdges)/step)):
            for x in range(int((imgShape[1] - scanEdges)/step)):
                scan = cv2.resize(img[step*y: step*y+scanEdges, step*x: step*x+scanEdges], dsize=(24, 24))
                key = (step *y, step *x)
                scanCollection[key] = scan

        posCollection = scanCollection
        tmpCollection = {}
        for i in range(self.cascadeNum):
            print("cascade {}, posCollection {} pieces".format(i+1, len(posCollection)))

            for scanP in posCollection.keys():
                prediction = self.ClassifierList[i].finalClassifier(posCollection[scanP])
                if prediction > 0:
                    tmpCollection[scanP] = posCollection[scanP]

            posCollection = tmpCollection
            tmpCollection = {}
            if len(posCollection) < creditableNum:
                break
        y = 0
        x = 0
        for point1 in posCollection.keys():
            y += point1[0]
            x += point1[1]
            #y = point1[0] + scanEdges
            #x = point1[1] + scanEdges
            #point2 = (y, x)
            #pt1 = (point1[1], point1[0])
            #pt2 = (point2[1], point2[0])
            #cv2.rectangle(img= inputImg, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=2)

        y = int(y/len(posCollection))
        x = int(x/len(posCollection))
        if x> y:
            y = x
        else:
            x = y

        pt1 = (x, y)
        pt2 = (x+scanEdges, y+scanEdges)
        cv2.rectangle(img=inputImg, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=2)
        imgShow(img=inputImg)

if __name__ == '__main__':
    cascadeNum =26
    maxIters = 20
    harr6Dir = load_dir(harr6Path)
    harr8Dir = load_dir(harr8Path)
    harr10Dir = load_dir(harr10Path)
    harr12Dir = load_dir(harr12Path)
    harr14Dir = load_dir(harr14Path)
    harr16Dir = load_dir(harr16Path)
    dir1 = select(harr16Dir, 0, 20)
    dir2 = select(harr14Dir, 0, 20)
    dir3 = select(harr14Dir, 20, 20)
    dir4 = select(harr12Dir, 0, 20)
    dir5 = select(harr12Dir, 20, 20)
    dir6 = select(harr12Dir, 40, 20)
    dir7 = select(harr10Dir, 0, 20)
    dir8 = select(harr10Dir, 20, 20)
    dir9 = select(harr10Dir, 40, 20)
    dir10 = select(harr10Dir, 60, 20)
    dir11 = select(harr8Dir, 0, 20)
    dir12 = select(harr8Dir, 20, 20)
    dir13 = select(harr8Dir, 40, 20)
    dir14 = select(harr8Dir, 60, 20)
    dir15 = select(harr8Dir, 80, 20)
    dir16 = select(harr8Dir, 100, 20)
    dir17 = select(harr8Dir, 120, 20)
    dir18 = select(harr6Dir, 0, 20)
    dir19 = select(harr6Dir, 20, 20)
    dir20 = select(harr6Dir, 40, 20)
    dir21 = select(harr6Dir, 60, 20)
    dir22 = select(harr6Dir, 80, 20)
    dir23 = select(harr6Dir, 100, 20)
    dir24 = select(harr6Dir, 120, 20)
    dir25 = select(harr6Dir, 140, 20)
    dir26 = select(harr6Dir, 160, 20)
    harrTempList = [dir1, dir2, dir3, dir4, dir5,
                    dir6, dir7, dir8, dir9, dir10,
                    dir11, dir12, dir13, dir14, dir15,
                    dir16, dir17, dir18, dir19, dir20,
                    dir21, dir22, dir23, dir24, dir25,
                    dir26]

    imgCollection = readImgCollection(posPath, negPath)
    HarrFeatures = HarrFeatures()
    CascadeClassifier = CascadeClassifier(harrTempList=harrTempList,
                                          StrongClassifier=strongClassifier,
                                          HarrFeatures=HarrFeatures,
                                          cascadeNum=cascadeNum,
                                          imgCollection=imgCollection,
                                          maxiters=maxIters,
                                          trained=True)

    CascadeClassifier.cascade()
    CascadeClassifier.predict(inputImg='../dataset/Putin/Putin_02.png', step = 8, scanEdges = 300, creditableNum=25)


