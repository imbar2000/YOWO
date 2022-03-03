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
#           list[[ class, score, np.array[[frame_index, box_index]] ]]
#           其中: frame_index 是det_result里的 帧下标
#                 box_index 是det_result里的 box下标

import os
import glob
import numpy as np
import pickle
import argparse


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
    return result

def build_tubes(det_result, th_tube):
    """
    根据检测结果, 生成tubes
    det_result  :检测结果
    th_tube     :分数阈值, 分数>th_tube的tube才会被留下来
    """
    tubes = []
    return tubes

def correct_yolov5(frm_result, frm_index, tubes):
    """
    用tube更正一帧的检测结果
    frm_result 是一帧的检测结果, list[frame_id, np.array[[x1, y1, x2, y2, cls0_score, cls1_score, cls2_score, class, obj_score]] ]
    """
    correct_box = {}    #被修改的box,  box_index: [class]
    for t in tubes:
        for f in t[TUBE_BOXES]:
            f_index, b_index = f
            if f_index == frm_index:
                # boxs = frm_result[FRM_BOXES]
                # boxs[b_index][BOX_CLSASS] = t[TUBE_CLASS]
                correct_box.get(b_index, default=[]).append(t[TUBE_CLASS: TUBE_SCORE+1])
                break
    
    for box_idx, t in correct_box.items():
        boxs = frm_result[FRM_BOXES]
        idx_tube = np.argmax(t, axis=1)
        cls = t[idx_tube][TUBE_CLASS]
        boxs[box_idx][BOX_CLSASS] = cls
        boxs[box_idx][BOX_OBJ_SCORE] = boxs[box_idx][BOX_CLS0_SCORE + cls]

def write_map_result(filName, objects):
    with open(filName, "w", encoding="utf-8") as fp:
        for obj in objects:
            x1, y1, x2, y2, _1, _2, _3, name, conf,  = obj
            fp.write("%s %.12f %d %d %d %d\n" % (name, conf, int(x1), int(y1), int(x2), int(y2)))

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
        correct_yolov5(det_result[i], i, tubes)

    save_mAP_result(det_result, output_dir, args.names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--yolov5_result_dir', default="/mnt/data/wangpeng/plate/test_plate_recg_v3.0_crop", help='path to detect results of yolov5')
    parser.add_argument('--output_dir', default="/mnt/data/wangpeng/plate/test_plate_recg_v3.0_crop", help='path to output results')
    parser.add_argument('--th_tube', default=0.3, type=float, help='threshod of tubes')
    parser.add_argument('--names', nargs='+', default=[], help='names of det result')
    parser.add_argument('--start_index_frm', type=int, default=10, help='begin build tubes')

    args = parser.parse_args()
    print(args)
    
    correct_yolov5_by_YOWO(args)

