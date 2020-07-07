import cv2
import numpy as np
from utils import imgShow, load_dir, normalization
import cProfile
from featureExtract import extract
from faceRe.KNN import KNNClassifier
from cascade import strongClassifier, cascadeClassifier, HarrFeatrues


# number 需要根据checkpoint中的number来设置
# rankNum
number = 15
rankNum = 30
posPath = '../data/samples/size_24_pt_90_nt_40_num_2/pos/'
negPath = '../data/samples/size_24_pt_90_nt_40_num_2/neg/'
intdictPath = '../checkpoints/intDir'
posIntDictName = 'size_24_pt_90_nt_40_num_2_pos'
negIntDictName = 'size_24_pt_90_nt_40_num_2_neg'
harrFeaturePath = '../checkpoints/harrFeatures/size_24_pt_90_nt_40_num_2'
alphaPath = '../checkpoints/classifierParam'
alphaName = 'classifier_alpha'
alphaName = alphaName + '_rank_' + str(rankNum) + '_num_' + str(number)
testimgPath = '../data/dataset/JohnMayer/JohnMayer_19.png'
classList = ["du jiang", "Eason", "Jay", "John Mayer", "Lin Yuner",
             "Merkel", "Obama", "Putin", "Xi Jinping", "Zhao Liying"]

if __name__ == '__main__':

    harr = HarrFeatrues(posPath=posPath, negPath=negPath,
                        intdictPath=intdictPath, harrFeaturePath=harrFeaturePath,
                        posIntDictName=posIntDictName, negIntDictName=negIntDictName,
                        if_intDictExist=True, cal_y2=False, if_y2=1)

    strongClassifier = strongClassifier(harr=harr, number=number, if_train=False,
                                        alphaPath=alphaPath, alphaName=alphaName)

    # 预测的人脸图片
    garyImg = strongClassifier.cascadePredict(img=testimgPath, cascadeNum=14, step=24, scanEdges=300, creditableNum=8)
    imgShow(img=garyImg)
    garyImg = normalization(cv2.resize(garyImg, (227, 227)))
    X_test = extract(if_predict=True, input=garyImg, paramPath='featureExtract/epoch__26_acc_0.85net_param.pkl')
    X_test = X_test.data.numpy().reshape(-1, 1)
    Y_train = np.array(load_dir('../data/trainFeature/Y_train.pkl')).reshape(1, -1)
    X_train = np.array(load_dir('../data/trainFeature/X_train.pkl')).transpose(1, 0)
    knn = KNNClassifier()
    Y_test = knn.model( X_test=X_test, X_train=X_train, Y_train=Y_train, k=3, print_correct=True)

    print("Predict result is:{}".format(classList[int(Y_test)]))
    print("done!")
