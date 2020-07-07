import numpy as np
import os
import cv2
from sklearn.decomposition import PCA
from .tools import extractClassname, extractFilename, normalization

pca = PCA(n_components=70, whiten=True)

classList = ["curry", "dujiang", "Eason", "FanBingbing", "Jay",
             "jingtian", "JohnMayer","kobe", "LinYuner", "Merkel",
             "Obama", "Putin", "XiJinping", "xuezhiqian", "ZhaoLiying"]

class KNNClassifier(object):

    def __init__(self):
        pass

    def distance(self, X_test, X_train):
        '''
        输入：
            X_test: 由numpy数组表示的测试集，大小为（图片长度 * 图片高度 *3， 测试样本）
            X_train: 由numpy数组表示的训练集，大小为（图片长度 * 图片高度 *3， 训练样本数）
        输出：
            distances: 测试数据与各个训练数据之间的距离，大小为（测试样本数量， 训练样本数量）的numpy数组
        '''
        #num_test = X_test.shape[1]
        #num_train = X_train[1]
        #distances = np.zeros((num_test, num_train))
        # 计算欧式距离
        # X与Y之间的欧氏距离：（X-Y）^2 = X^2 - 2XY + Y^2
        dist1_2XY = np.multiply(np.dot(X_test.T, X_train), -2)
        dist2_X2 = np.sum(np.square(X_test.T), axis=1, keepdims=True)
        dist3_Y2 = np.sum(np.square(X_train), axis=0, keepdims=True)
        distances = np.sqrt((dist1_2XY + dist2_X2 + dist3_Y2)+0.001)

        return distances

    def predict(self, X_test, X_train, Y_train, k=1):
        '''
        输入：
            X_test: 由numpy数组表示的测试集， 大小为（图片长度 * 图片高度 * 3， 测试样本数目）
            X_train: 由numpy数组表示的训练集，大小为（图片长度 * 图片高度 * 3， 训练样本数目）
            Y_train: 由numpy数组表示的训练标签， 大小为（1， 训练样本数）
            k: 选取的最近邻个数
        输出：
            Y_prediction: 包含X_test种所有的而预测值的numpy数组（向量）
            distances: 由numpy数组表示的测试数据与各个训练数据之间的距离， 大小为（测试样本数， 训练样本数）
        '''
        distances = self.distance(X_test, X_train)
        num_test = X_test.shape[1]
        Y_prediction = np.zeros(num_test)
        for i in range(num_test):
            # 将测试数据与所有训练数据的距离递增排序，选取距离最小的k个点, 并返回索引
            cur_distance = distances[i, :]
            min_k_index = np.argsort(cur_distance)[:k]
            # 读取k个最近邻的类别信息
            y_labels_k  = Y_train[0][min_k_index]
            # 返回前k个点中出现频率最高的类别
            # 由于有num_test个测试数据， 因此该变量为[num_test]的向量
            # [类别0的个数， 类别1的个数， ..., 类别n的个数]
            class_count = np.bincount(y_labels_k)
            Y_prediction[i] = np.argmax(class_count)

        return Y_prediction, distances

    def model(self, X_test=None, Y_test=None, X_train=None, Y_train=None, k=1, print_correct=False, if_test=True):
        '''
        输入：
            X_test: 由numpy数组表示的测试集， 大小为（图片长度 * 图片高度 * 3， 测试样本数目）
            X_train: 由numpy数组表示的训练集，大小为（图片长度 * 图片高度 * 3， 训练样本数目）
            Y_train: 由numpy数组表示的训练标签， 大小为（1， 训练样本数）
            Y_test: 由numpy数组表示的测试数据，大小为（1， 测试样本数）
            k: 选取的最近邻个数
            print_correct ：是否打印测试正确率
        输出：
            包含模型信息的字典
        '''
        Y_prediction, distances = self.predict(X_test, X_train, Y_train, k)
        # 无标签图片的预测
        if if_test:
            return Y_prediction

        num_correct = np.sum(Y_prediction == Y_test)
        accuracy = np.mean(Y_prediction == Y_test)
        if print_correct:
            print("Correct:{}/{}, the test accuracy is:{}".format(num_correct, np.shape(X_test)[1], accuracy))
        d = {"k": k,
             "Y_prediction":Y_prediction,
             "distances":distances,
             "accuracy":accuracy}

        return d

    def readSet(self, path = None, if_print=True):
        X = []
        Y = []
        for f in os.listdir(path):
            # 依次读入图片， 将图片转化为灰度图片并标准化后拉伸成一维数据
            imgName = path + f
            grayImg = cv2.cvtColor(cv2.imread(imgName), cv2.COLOR_BGR2GRAY)
            normImg = normalization(grayImg)
            #pca.fit(normImg)
            #normImg = pca.fit_transform(normImg)
            X.append(np.reshape(normImg, -1))
            tmp_filename = extractFilename(f)
            train_class = extractClassname(tmp_filename)

            # 读入数据集的标签
            if train_class == classList[0]:
                Y.append(0)
            elif train_class == classList[1]:
                Y.append(1)
            elif train_class == classList[2]:
                Y.append(2)
            elif train_class == classList[3]:
                Y.append(3)
            elif train_class == classList[4]:
                Y.append(4)
            elif train_class == classList[5]:
                Y.append(5)
            elif train_class == classList[6]:
                Y.append(6)
            elif train_class == classList[7]:
                Y.append(7)
            elif train_class == classList[8]:
                Y.append(8)
            elif train_class == classList[9]:
                Y.append(9)
            elif train_class == classList[10]:
                Y.append(10)
            elif train_class == classList[11]:
                Y.append(11)
            elif train_class == classList[12]:
                Y.append(12)
            elif train_class == classList[13]:
                Y.append(13)
            elif train_class == classList[14]:
                Y.append(14)

        X = np.array(X)
        pca.fit(X)
        X_pca = pca.fit_transform(X)

        if if_print == True:
            print(pca.explained_variance_ratio_)
            print("total :", np.sum(pca.explained_variance_ratio_))
            #imgShow(img=newImg)
            print("mod:", np.linalg.norm(X_pca, ord=2))

        X = X_pca.transpose(1, 0)
        Y = np.array(Y).reshape(1, -1)

        return X, Y

if __name__ == '__main__':
    train_path = '../samples/testImg/train/'
    test_path = '../samples/testImg/test/'
    knn = KNNClassifier()
    X_train, Y_train = knn.readSet(train_path)
    X_test, Y_test = knn.readSet(test_path)
    knn.model(X_test, Y_test, X_train, Y_train, k=3, print_correct= True)
    print("done!")