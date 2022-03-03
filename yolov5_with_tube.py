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
#           list[class, score, [[frame_index, box_index]]]
#           其中: frame_index 是det_result里的 帧下标
#                 box_index 是det_result里的 box下标

import os
import glob
import numpy as np
import pickle
import argparse


def read_yolov5_result(result_dir):
    result = []
    return result

def build_tubes(frm_result):
    """
        根据检测结果, 生成tubes
    """
    tubes = []
    return tubes

def correct_yolov5(frm_result, tubes):
    """
        用tube更正一帧的检测结果
    """
    pass

def save_mAP_result(det_result, output_dir):
    pass


def correct_yolov5_by_YOWO(args):
    yolov5_result_dir, output_dir = args.yolov5_result_dir, args.output_dir
    det_result = read_yolov5_result(yolov5_result_dir)

    # 从min_frms开始处理每一帧
    min_frms = 10
    for i in range(min_frms, len(det_result)):
        tubes = build_tubes(det_result[:i+1])
        correct_yolov5(det_result[i], tubes)

    save_mAP_result(det_result, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--yolov5_result_dir', default="/mnt/data/wangpeng/plate/test_plate_recg_v3.0_crop", help='path to detect results of yolov5')
    parser.add_argument('--output_dir', default="/mnt/data/wangpeng/plate/test_plate_recg_v3.0_crop", help='path to output results')

    args = parser.parse_args()
    print(args)
    
    correct_yolov5_by_YOWO(args.yolov5_result_dir, args.output_dir)

