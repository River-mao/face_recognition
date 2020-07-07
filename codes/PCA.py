# Author: River Mao
# 主成分分析算法实现图片降维
# 主成分分析， 是一种数据降维技术， 用于数据处理
# 通过PCA降维不仅可以减少计算量还可以滤除部分噪声

import cv2
import numpy as np
from sklearn.decomposition import PCA
from tool import imgShow

if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread("../samples/size_64_t_20to40_70_num_20/pos/curry1.png"), cv2.COLOR_BGR2GRAY)
    pca = PCA(n_components=32)
    pca.fit(img)
    newImg = pca.fit_transform(img)

    print(pca.explained_variance_ratio_)
    print("total :",np.sum(pca.explained_variance_ratio_))
    imgShow( img = newImg)
    print("mod:",np.linalg.norm(newImg, ord=2))
