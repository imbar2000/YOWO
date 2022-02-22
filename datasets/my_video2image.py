# -*- coding: utf-8 -*-
# ucf数据集, 视频转图片
import cv2
import os

def video_to_images(video_dir, output_dir):
    videos = os.listdir(video_dir)
    for j,video in enumerate(videos):
        video_path = os.path.join(video_dir, video)
        input_movie = cv2.VideoCapture(video_path)
        print('%s, progress=%d/%d'%(video, j, len(videos)))
        i = 1
        while True:
            ret, frame = input_movie.read()
            if not ret:
                # print("read fail, i=%d" % i)
                break

            className = video.split('_')[1]
            videoName = video[:-4]
            jpgPath = os.path.join(output_dir, className, videoName)
            if not os.path.exists(jpgPath):
                os.makedirs(jpgPath)
            jpgName = os.path.join(jpgPath, "%05d.jpg" % i)
            i = i + 1
            # print(jpgName)
            cv2.imwrite(jpgName, frame)
        # break

if __name__ == "__main__":
    video_dir = "/dataset/YSTData/015_action_recg/ucf24/test/video"
    output_dir = "/dataset/YSTData/015_action_recg/ucf24/test/rgb-images"
    video_to_images(video_dir, output_dir)


