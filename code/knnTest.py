import numpy as np
from faceRe.KNN import KNNClassifier
from utils import imgShow, load_dir

if __name__ == '__main__':

    Y_train = np.array(load_dir('../data/trainFeature/Y_train.pkl')).reshape(1, -1)
    X_train = np.array(load_dir('../data/trainFeature/X_train.pkl')).transpose(1, 0)
    Y_test = np.array(load_dir('../data/trainFeature/Y_test.pkl')).reshape(1, -1)
    X_test = np.array(load_dir('../data/trainFeature/X_test.pkl')).transpose(1, 0)
    knn = KNNClassifier()
    Y_test = knn.model( X_test=X_test, Y_test=Y_test,
                        X_train=X_train, Y_train=Y_train,
                        k=3, print_correct=True, if_test=False)

