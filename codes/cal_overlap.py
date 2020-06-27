# Author: River Mao

import cv2
import numpy as np

def cal_overlap(ROI1P, ROI2P, grayImg):
    '''
    计算同一张图片上的两个区域的重叠率
    input:
        ROI1P: 区域一的坐标位置，[y, x, height, width]，输入时默认第一位位置的ROI为正样本区
        ROI2P: 区域二的坐标，[y, x, height, width]，输入时设置为待检测的区域，默认ROI1P和ROI2P大小相同
        grayImg：灰度图像
    output:
        重叠率：重叠区域/兴趣区
    '''

    height, width = np.shape(grayImg)
    ROI1 = np.zeros((height, width))
    ROI2 = np.zeros((height, width))
    ROI1y1, ROI1x1 = ROI1P[0],ROI1P[1]
    ROI1y2, ROI1x2 = ROI1y1 + ROI1P[2],  ROI1x1 + ROI1P[3]
    ROI2y1, ROI2x1 = ROI2P[0], ROI2P[1]
    ROI2y2, ROI2x2 = ROI2y1 + ROI2P[2], ROI2x1 + ROI2P[3]
    ROI1[ROI1y1:ROI1y2, ROI1x1:ROI1x2] = 255
    ROI2[ROI2y1:ROI2y2, ROI2x1:ROI2x2] = 255
    #取消注释显示掩膜
    #cv2.namedWindow('ROI1',cv2.WINDOW_NORMAL)
    #cv2.imshow('ROI1', ROI1)
    #cv2.waitKey(0)
    #cv2.namedWindow('ROI2',cv2.WINDOW_NORMAL)
    #cv2.imshow('ROI2', ROI2)
    #cv2.waitKey(0)
    overlapArea = ROI1 * ROI2
    overlapRatio = np.count_nonzero(overlapArea)/(ROI1P[2]* ROI1P[3])

    return overlapRatio


# 测试
if __name__ == '__main__':
    grayImg = cv2.cvtColor(cv2.imread('testImg.png'), cv2.COLOR_BGR2GRAY)
    ROI1P = [0,0,100,100]
    ROI2P = [50,50,100,100]
    overlapRatio = cal_overlap(ROI1P, ROI2P,grayImg)
    print("the overlap ratio is:", overlapRatio)
