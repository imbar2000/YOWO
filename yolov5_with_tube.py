# 用YOWO里的tube, 来更正yolov5的box标签
#   步骤: 1 读取yolov5检测结果
#         2 构建tube
#         3 根据tube更正yolov5标签
#         4 更正后的结果保存成mAP格式的文件
#
#   数据结构
#       目标检测结果 det_result:
#           list[[frame_id, np.array[[x1, y1, x2, y2, cls0_score, cls1_score, cls2_score, class, obj_score]] ]]
#           其中: 
#                 frame_id是帧号, 注意: 它不是list的下标
#                 cls0_score, cls1_score, cls2_score是三个类别的分数
#                 class是类别
#                 obj_score是class这个类别的分数. 也是三个分数里最高的那个
#        tubes: 
#           list[[ class, tube_score, np.array[[frame_id, x1, y1, x2, y2, cls_score]] ]]

import os
import glob
from matplotlib.pyplot import box
import numpy as np
import pickle
import argparse
from core.eval_results import link_video_one_class

CLASS_NUM = 3

BOX_X1, BOX_Y1, BOX_X2, BOX_Y2 = 0, 1, 2, 3
BOX_CLS0_SCORE = 4
BOX_CLSASS = 7
BOX_OBJ_SCORE = 8

FRM_INDEX = 0
FRM_BOXES = 1

TUBE_CLASS = 0
TUBE_SCORE = 1
TUBE_BOXES = 2

def read_yolov5_result(result_dir):
    result = []

    files = os.listdir(result_dir) # 得到文件夹下的所有文件名称
    print(files)

    files.sort()
    files.sort(key = lambda x:int(x[:-4])) # 文件名按数字排序
    print(files)

    CLASSES = ('cigarette', 'phone', 'other')
    for i in range(len(CLASSES)):
        print('class[%d] : ' %i, CLASSES[i])

    box = np.zeros((1,5), dtype = float)

    frame_data = []
    count = 0 # 作为frame_id
    for file in files: # 遍历文件夹
        if not os.path.isdir(file): # 判断是否是文件夹，不是文件夹才打开
            count = count + 1
            data_index = 0
            print('count = ', count)
            with open(result_dir + '/' + file, "r") as f:  # 打开文件
                file_data = f.readlines()
                file_line_num = len(file_data)
                print('file_line_num = ', file_line_num)
                file_data_list = []
                for no_line_data in file_data:
                    no_line_data = no_line_data.strip('\n')  # 去掉列表中每一个元素的换行符
                    data_list = no_line_data.split(' ') # 字符串转list
                    data_list[0:7] = list(map(float, data_list[0:7])) # 字符转数字
                    print('%s : ' %file, data_list)
                    file_data_list.append(data_list)
                    data_index = data_index + 1
                boxes = np.zeros((file_line_num, 9))
                for j in range(file_line_num):
                    boxes[j,:4] = file_data_list[j][3:7]
                    boxes[j,4:7] = file_data_list[j][0:3]
                    boxes[j,7] = file_data_list[j][0:3].index(max(file_data_list[j][0:3]))
                    boxes[j,8] = max(file_data_list[j][0:3])
                print('boxes =', boxes)
            frame_data.append([count, boxes])

    # return result
    return frame_data


def build_yowo_class_result(det_result, cls):
    """
    检测结果转成yolwo的单类别格式
    返回: list[ [frame_index, np.array[[x1, y1, x2, y2, cls_score]] ] ]
    """
    yowo_cls_result = []
    index = [0, 1, 2, 3, BOX_CLS0_SCORE+cls]
    for fr in det_result:
        yr = [fr[0], fr[1][index]]
        yowo_cls_result.append(yr)
    return yowo_cls_result

def build_tubes(det_result, th_tube):
    """
    根据检测结果, 生成tubes
    det_result  :检测结果
    th_tube     :分数阈值, 分数>th_tube的tube才会被留下来
    返回        :tubes, list[[ class, tube_score, np.array[[frame_id, x1, y1, x2, y2, cls_score]] ]]
    """
    # 生成yowo格式tubes
    tubes = []
    for k in range(CLASS_NUM):
        yowo_cls_result = build_yowo_class_result(det_result, k)
        pred_link_v = link_video_one_class(yowo_cls_result, bNMS3d=True, gtlen=None) 
        video_index = 0
        for tube in pred_link_v:
            # video_index = video_index + 1
            # pred.append((video_index, tube))
            tube_scores = [np.mean(b[:, 5]) for b in tube]
            if tube_scores > th_tube:
                tubes.append([k, tube_scores, tube])
    return tubes

def find_box(frm_result, tube_box):
    """
    frm_result里查找tube_box对应的box下标
    """
    for i, b in enumerate(frm_result[FRM_BOXES]):
        if tube_box[1:5] == b[:4]:      # x1,y1,x2,y2是否相等
            return i
    return -1

def correct_yolov5(frm_result, tubes):
    """
    用tube更正一帧的检测结果
    frm_result 是一帧的检测结果, list[frame_id, np.array[[x1, y1, x2, y2, cls0_score, cls1_score, cls2_score, class, obj_score]] ]
    tubes: list[[ class, tube_score, np.array[[frame_id, x1, y1, x2, y2, cls_score]] ]]
    """
    correct_box = {}    #被修改的box,  box_index: np.array[[class, tube_score]]
    for t in tubes:
        t_boxes = t[TUBE_BOXES][-1]     #仅使用tube的最后一个box
        if t_boxes[0] == frm_result[FRM_INDEX]:
            box_index = find_box(frm_result, t_boxes)
            if box_index >= 0:
                correct_box.get(box_index, default=[]).append(t[TUBE_CLASS: TUBE_SCORE+1])
                break
    
    for bi, t in correct_box.items():
        boxs = frm_result[FRM_BOXES]
        idx_tube = np.argmax(t, axis=1)
        cls = t[idx_tube][TUBE_CLASS]
        boxs[bi][BOX_CLSASS] = cls
        boxs[bi][BOX_OBJ_SCORE] = boxs[bi][BOX_CLS0_SCORE + cls]

def save_mAP_result(det_result, output_dir, names):
    """
    保存成map格式
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for d in det_result:
        fname = os.path.join(output_dir, "%4d.txt" % d[FRM_INDEX])
        with open(fname, "w", encoding="utf-8") as fp:
            for box in d[FRM_BOXES]:
                fp.write("%s %.12f %d %d %d %d\n" % (names[box[BOX_CLSASS]],  box[BOX_OBJ_SCORE], int(box[BOX_X1]), int(box[BOX_Y1]), int(box[BOX_X2]), int(box[BOX_Y2])))


def correct_yolov5_by_YOWO(args):
    yolov5_result_dir, output_dir = args.yolov5_result_dir, args.output_dir
    det_result = read_yolov5_result(yolov5_result_dir)

    # 从min_frms开始处理每一帧
    min_frms = args.start_index_frm
    for i in range(min_frms, len(det_result)):
        tubes = build_tubes(det_result[:i+1], args.th_tube)
        correct_yolov5(det_result[i], tubes)

    save_mAP_result(det_result, output_dir, args.names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--yolov5_result_dir', default="/dataset/wanghaifeng/read-yolov5-txt/labels", help='path to detect results of yolov5')
    parser.add_argument('--output_dir', default="/mnt/data/wangpeng/plate/test_plate_recg_v3.0_crop", help='path to output results')
    parser.add_argument('--th_tube', default=0.3, type=float, help='threshod of tubes')
    parser.add_argument('--names', nargs='+', default=[], help='names of det result')
    parser.add_argument('--start_index_frm', type=int, default=10, help='begin build tubes')

    args = parser.parse_args()
    print(args)
    
    correct_yolov5_by_YOWO(args)

