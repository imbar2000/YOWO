# -*- coding: utf-8 -*-
# 标注文件名有两种格式:
# 格式1: groundtruths_ucf.zip的标注文件名,例如 Diving_v_Diving_g04_c05_00042.txt
# 格式2: testlist.txt的格式, 例如 Basketball/v_Basketball_g01_c01/00009.txt
# 本脚本 把格式1 改成格式2

import os
import shutil

def trans_label_fname(input_dir, output_dir):
    label_files = os.listdir(input_dir)
    for j, f in enumerate(label_files):
        slice = f.split('_')
        if len(slice) != 6:
            print("parse fname fail:%s" % f)
            break
        className, *videoName, fname = slice
        path = os.path.join(output_dir, className, "%s_%s_%s_%s" % (videoName[0], videoName[1], videoName[2], videoName[3]))
        #print(className, videoName, fname)
        if not os.path.exists(path):
            os.makedirs(path)

        dst_name = os.path.join(path, fname)
        src_name = os.path.join(input_dir, f)
        # print(src_name, dst_name)
        shutil.move(src_name, dst_name)

        if j % 1000 == 0:
            print("progress:%d/%d" % (j, len(label_files)))
        

if __name__ == "__main__":
    input_dir = "/dataset/YSTData/015_action_recg/ucf24/test/groundtruths_ucf"
    output_dir = "/dataset/YSTData/015_action_recg/ucf24/test/label"
    trans_label_fname(input_dir, output_dir)


