# Author: River Mao
import os
import cv2
import numpy as np

current_pos = None
tl = None
br = None

#鼠标事件
def get_rect(im, title='get_rect'):   #   (a,b) = get_rect(im, title='get_rect')
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,
        'released_once': False}

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):

        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)
        if mouse_params['tl'] is not None:
            y0, x0 = mouse_params['tl'][0], mouse_params['tl'][1]
            tl = (y0, x0)

            y1, x1 = mouse_params['current_pos'][0], mouse_params['current_pos'][1]
            width = x1 - x0
            height = y1 - y0
            if height > width:
                cv2.rectangle(im_draw, mouse_params['tl'], (y0+height, x0+height), (255, 0, 0))
                br = (y0 + height, x0 + height)
            else:
                cv2.rectangle(im_draw, mouse_params['tl'], (y0 + width, x0 + width), (255, 0, 0))
                br = (y0 + width, x0 + width)

        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)

    cv2.destroyWindow(title)

    return (tl, br)  #tl=(y1,x1), br=(y2,x2)

def resize_and_label(img_Dir, ori_path, save_path, spilt, class_name, extension):
    if not os.path.exists(save_path +class_name):
        os.mkdir(save_path+class_name)

    for i in range(len(img_Dir)):
        img = cv2.imread(ori_path + img_Dir[i])
        img = cv2.resize(img, dsize=(600, 800))
        (a, b) = get_rect(img, title='get_rect')
        y_start = a[1]
        x_start = a[0]
        height = b[1] - a[1]
        width =  b[0] - a[0]
        file_name =  class_name +'_'+ "%02d" % i + "." + extension
        save_file_name = save_path + class_name+'/' + file_name
        cv2.imwrite(filename=save_file_name, img=img)

        label =class_name +'/' + file_name +\
               spilt +str(x_start)+ spilt +str(y_start)+ \
               spilt +str(height)+ spilt +str(width) +'\n'

        label_file = save_path + class_name + '/' + class_name + '.txt'
        f = open(label_file, 'a', encoding='utf-8')
        f.write(label)
        f.close()

if __name__ == '__main__':
    # 原文件路径
    ori_path = "./ZhaoLiying/"
    img_Dir = os.listdir(ori_path)
    save_path = '../dataset/'
    spilt = ','
    class_name = 'ZhaoLiying'
    extension = 'png'
    resize_and_label(img_Dir, ori_path, save_path, spilt, class_name, extension)

