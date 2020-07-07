from utils import readImgCollection, normalization, load_dir, save_dir, imgShow
from cascade import strongClassifier, cascadeClassifier, HarrFeatrues
import cv2
import time
import numpy as np

if __name__ == '__main__':

    rankNum = 30
    number = 15
    posPath = '../data/samples/size_24_pt_90_nt_40_num_2/pos/'
    negPath = '../data/samples/size_24_pt_90_nt_40_num_2/neg/'
    intdictPath = '../checkpoints/intDir'
    posIntDictName = 'size_24_pt_90_nt_40_num_2_pos'
    negIntDictName = 'size_24_pt_90_nt_40_num_2_neg'
    harrFeaturePath = '../checkpoints/harrFeatures/size_24_pt_90_nt_40_num_2'
    alphaPath = '../checkpoints/classifierParam'
    alphaName = 'classifier_alpha'
    testimgPath = '../data/samples/size_24_pt_90_nt_40_num_2/pos/dujiang_18_2p.png'

    # 储存了图片名和标签的字典, 包括了父文件夹posPath, negPath
    imgCollection = readImgCollection(posPath=posPath, negPath=negPath)

    harr = HarrFeatrues(posPath=posPath, negPath=negPath,
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
        Classifier.boosting(rank=rank, imgCollection=imgCollection)

    testImg = normalization(cv2.imread(testimgPath, cv2.COLOR_BGR2GRAY))

    predicttion = Classifier.finalClassifier(testImg)
    print("the prediction is:", predicttion)
    print("Done!")
