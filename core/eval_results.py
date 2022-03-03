# -*- coding:utf-8 -*-
import numpy as np
import os
from core.utils import *
import time

def compute_score_one_class(bbox1, bbox2, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbx: <x1> <y1> <x2> <y2> <class score>
    n_bbox1 = bbox1.shape[0]
    n_bbox2 = bbox2.shape[0]
    # for saving all possible scores between each two bbxes in successive frames
    scores = np.zeros([n_bbox1, n_bbox2], dtype=np.float32)
    for i in range(n_bbox1):
        box1 = bbox1[i, :4]
        for j in range(n_bbox2):
            box2 = bbox2[j, :4]
            bbox_iou_frames = bbox_iou(box1, box2, x1y1x2y2=True)
            sum_score_frames = bbox1[i, 4] + bbox2[j, 4]
            mul_score_frames = bbox1[i, 4] * bbox2[j, 4]
            scores[i, j] = w_iou * bbox_iou_frames + w_scores * sum_score_frames + w_scores_mul * mul_score_frames

    return scores

def link_bbxes_between_frames(bbox_list, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbox_list: 每帧的检测结果, list[ np.array[[x1, y1, x2, y2, cls_score]] ] , 注意每帧的box数都不一样
    # check no empty detections
    ind_notempty = []
    nfr = len(bbox_list)
    for i in range(nfr):
        if np.array(bbox_list[i]).size:
            ind_notempty.append(i)
    # 如果没有检测框, 返回空
    if not ind_notempty:
        return []
    # 如果某些帧没有检测框, 从最近的帧里复制一份检测框
    elif len(ind_notempty)!=nfr:     
        for i in range(nfr):
            if not np.array(bbox_list[i]).size:
                # copy the nearest detections to fill in the missing frames
                ind_dis = np.abs(np.array(ind_notempty) - i)
                nn = np.argmin(ind_dis)
                bbox_list[i] = bbox_list[ind_notempty[nn]]

    
    detect = bbox_list
    nframes = len(detect)
    res = []
    
    # 计算i帧的box 与i+1的 边的分数
    #           帧0     帧1     帧2     帧3
    # box数     4       3       1       2
    # 边的shape (4,3)   (3,1)   (1,2)
    is_frame_empty = np.zeros([nframes,], dtype=np.bool)
    edge_scores = [compute_score_one_class(detect[i], detect[i+1], w_iou=w_iou, w_scores=w_scores, w_scores_mul=w_scores_mul) for i in range(nframes-1)]
    copy_edge_scores = edge_scores

    while not np.any(is_frame_empty):   # while( 所有帧都不为空 ). 因为有多个tube的情况, 所以这里需要循环
        # initialize
        scores = [np.zeros([d.shape[0],], dtype=np.float32) for d in detect]
        index = [np.nan*np.ones([d.shape[0],], dtype=np.float32) for d in detect]
        # viterbi
        # 从倒数第2到第1个, 统计最大的分数 
        for i in range(nframes-2, -1, -1):
            edge_score = edge_scores[i] + scores[i+1]   # 第i帧边分数 加 i+1帧顶点的分数
            # find the maximum score for each bbox in the i-th frame and the corresponding index
            scores[i] = np.max(edge_score, axis=1)      # 更新第i帧的顶点分数, 即从它开始所有路径的最大分数
            index[i] = np.argmax(edge_score, axis=1)    # 第i帧, 各个顶点的最大相邻顶点
        
        # decode. 建立路径
        idx = -np.ones([nframes], dtype=np.int32)
        idx[0] = np.argmax(scores[0])       # 第0帧的最优顶点
        for i in range(0, nframes-1):       
            idx[i+1] = index[i][idx[i]]     # 第i+1帧的最优顶点 = 第i帧 最优顶点(idx[i]) 的最大相邻顶点

        # 删除已经覆盖的boxes, 并且建立输出结构体
        # 这里把已建立tube的所有顶点都删除了, 是不是有问题?
        # 应该允许多个tube经过同一个中间顶点, 但不允许多个tube有一个起始顶点 
        # 是不是应该只删除第0帧的, tube的顶点
        this = np.empty((nframes, 6), dtype=np.float32)
        this[:, 0] = 1 + np.arange(nframes)
        for i in range(nframes):
            j = idx[i]
            iouscore = 0
            if i < nframes-1:
                iouscore = copy_edge_scores[i][j, idx[i+1]] - bbox_list[i][j, 4] - bbox_list[i+1][idx[i+1], 4]

            # 删除顶点: 第i帧, 第j个顶点;
            if i < nframes-1:   # 如果不是最后一帧, 删除该顶点右边的edge
                edge_scores[i] = np.delete(edge_scores[i], j, 0)
            if i > 0:           # 如果不是第一帧, 删除该顶点左边的edge
                edge_scores[i-1] = np.delete(edge_scores[i-1], j, 1)
            this[i, 1:5] = detect[i][j, :4]
            this[i, 5] = detect[i][j, 4]
            detect[i] = np.delete(detect[i], j, 0)  # 删除detect box
            is_frame_empty[i] = (detect[i].size==0) # 更新第i帧是否空的标志
        res.append( this )
        if len(res) == 3:   # 最多输出3个tube
            break
        
    return res


def link_video_one_class(vid_det, bNMS3d = True, gtlen=None):
    '''
    在一个视频里, 把一个类别的box连起来, 形成tube
    vid_det: 每帧的检测结果, list[ [frame_index, np.array[[x1, y1, x2, y2, cls_score]] ] ]
    gtlen: the mean length of gt in training set
    return: list[ np.array[[frame_index, x1, y1, x2, y2, cls_score]] ]
    '''
    # list of bbox information [[bbox in frame 1], [bbox in frame 2], ...]
    vdets = [vid_det[i][1] for i in range(len(vid_det))]
    vres = link_bbxes_between_frames(vdets) 
    if len(vres) != 0:
        if bNMS3d:
            tube = [b[:, :5] for b in vres]
            # compute score for each tube
            tube_scores = [np.mean(b[:, 5]) for b in vres]
            dets = [(tube[t], tube_scores[t]) for t in range(len(tube))]
            # nms for tubes
            keep = nms_3d(dets, 0.3) # bug for nms3dt
            if np.array(keep).size:
                vres_keep = [vres[k] for k in keep]
                # max subarray with penalization -|Lc-L|/Lc
                if gtlen:
                    vres = temporal_check(vres_keep, gtlen)
                else:
                    vres = vres_keep

    return vres

def summary_videos(pred_videos):
    cnt_video, cnt_frame, cnt_box = len(pred_videos), 0, 0
    for v in pred_videos:
        cnt_frame = cnt_frame + len(v[1])
        for f in v[1]:
            cnt_box = cnt_box + len(f[1])
    return cnt_video, cnt_frame, cnt_box

def video_ap_one_class(gt, pred_videos, iou_thresh = 0.2, bTemporal = False, gtlen = None):
    '''
    gt:          [[video_index, np.array[[frame_index, x1, y1, x2, y2]] ] ]
    pred_videos: [[video_index, [[frame_index, np.array[[x1, y1, x2, y2, cls_score]] ]] ]]
        gt里, 每个视频多个帧, 每帧只有一个box;
        pred_videos里, 帧数和gt一样, 但每帧里有多个框. 需要从每帧里找一个框,串起来,形成一条路径,是动态路径规划的问题. 用维特比算法来解
    '''

    t0 = time.time()

    # link for prediction
    pred = []
    for pred_v in pred_videos:      # 遍历每个视频
        video_index = pred_v[0]

        # 连接成tube, [ [frame_index, x1, y1, x2, y2, cls_score] ]
        pred_link_v = link_video_one_class(pred_v[1], bNMS3d=True, gtlen=gtlen)
        for tube in pred_link_v:
            pred.append((video_index, tube))

    t1 = time.time()
    cnt_video, cnt_frame, cnt_box = summary_videos(pred_videos)
    print("        build tube: %f sec, cnt_video=%d, cnt_frame=%d, cnt_box=%d" % (t1-t0, cnt_video, cnt_frame, cnt_box))

    # pred的tube 按降序排序
    # sort tubes according to scores (descending order)
    argsort_scores = np.argsort(-np.array([np.mean(b[:, 5]) for _, b in pred])) 
    pr = np.empty((len(pred)+1, 2), dtype=np.float32) # precision, recall
    pr[0,0] = 1.0
    pr[0,1] = 0.0
    fn = len(gt) #sum([len(a[1]) for a in gt])
    fp = 0
    tp = 0

    # 计算pred的tube的ap值
    gt_v_index = [g[0] for g in gt]
    for i, k in enumerate(argsort_scores):
        # if i % 100 == 0:
        #     print ("%6.2f%% boxes processed, %d positives found, %d remain" %(100*float(i)/argsort_scores.size, tp, fn))
        video_index, boxes = pred[k]
        ispositive = False
        if video_index in gt_v_index:
            gt_this_index, gt_this = [], []
            for j, g in enumerate(gt):
                if g[0] == video_index:
                    gt_this.append(g[1])
                    gt_this_index.append(j)
            if len(gt_this) > 0:
                if bTemporal:
                    iou = np.array([iou3dt(np.array(g), boxes[:, :5]) for g in gt_this])
                else:            
                    if boxes.shape[0] > gt_this[0].shape[0]:
                        # in case some frame don't have gt 
                        iou = np.array([iou3d(g, boxes[int(g[0,0]-1):int(g[-1,0]),:5]) for g in gt_this]) 
                    elif boxes.shape[0]<gt_this[0].shape[0]:
                        # in flow case 
                        iou = np.array([iou3d(g[int(boxes[0,0]-1):int(boxes[-1,0]),:], boxes[:,:5]) for g in gt_this]) 
                    else:
                        iou = np.array([iou3d(g, boxes[:,:5]) for g in gt_this]) 

                if iou.size > 0: # on ucf101 if invalid annotation ....
                    argmax = np.argmax(iou)
                    if iou[argmax] >= iou_thresh:
                        ispositive = True
                        del gt[gt_this_index[argmax]]
        if ispositive:
            tp += 1
            fn -= 1
        else:
            fp += 1
        pr[i+1,0] = float(tp)/float(tp+fp)
        pr[i+1,1] = float(tp)/float(tp+fn + 0.00001)
    ap = voc_ap(pr)

    return ap


def gt_to_videts(gt_v):
    # return  [label, video_index, [[frame_index, x1,y1,x2,y2], [], []] ]
    keys = list(gt_v.keys())
    keys.sort()
    res = []
    for i in range(len(keys)):
        # annotation of the video: tubes and gt_classes
        v_annot = gt_v[keys[i]]
        for j in range(len(v_annot['tubes'])):
            res.append([v_annot['gt_classes'], i+1, v_annot['tubes'][j]])
    return res


def evaluate_videoAP(gt_videos, all_boxes, CLASSES, iou_thresh = 0.2, bTemporal = False, prior_length = None):
    '''
    gt_videos : {'video_name': {'gt_classes':1, 'tubes':[[frame_index, x1,y1,x2,y2]]} }
    all_boxes: {"jpg_name": { class: np.array[[x1, y1, x2, y2, cls_score]]} }
                有24个类别的class
    '''
    def imagebox_to_videts(img_boxes, CLASSES):
        # image names
        keys = list(all_boxes.keys())
        keys.sort()
        res = []                                            # 返回值
        # without 'background'
        for cls_ind, cls in enumerate(CLASSES[0:]):         # 遍历24个类别
            v_cnt = 1
            frame_index = 1
            v_dets = []
            cls_ind += 1
            # get the directory path of images
            preVideo = os.path.dirname(keys[0])                 # 前一个视频名称
            for i in range(len(keys)):                      # 遍历每一帧
                curVideo = os.path.dirname(keys[i])
                img_cls_dets = img_boxes[keys[i]][cls_ind]      # 该帧该类别的检测结果, shape=(nbox, 5)
                v_dets.append([frame_index, img_cls_dets])      # 格式[frame_index, [x1, y1, x2, y2, cls_score]]
                frame_index += 1
                if preVideo!=curVideo:                          
                    preVideo = curVideo                         # 到下个视频文件了, 需要保存前一个视频文件的数据
                    frame_index = 1
                    # tmp_dets = v_dets[-1]
                    del v_dets[-1]                              # v_dets[-1]是新视频文件的, 其余是旧视频的, 所以删掉
                    res.append([cls_ind, v_cnt, v_dets])        # append到res, 格式[cls_id, 视频索引, v_dets(检测结果)]
                    v_cnt += 1
                    v_dets = []
                    # v_dets.append(tmp_dets)
                    v_dets.append([frame_index, img_cls_dets])
                    frame_index += 1
            # the last video
            # print('num of videos:{}'.format(v_cnt))
            res.append([cls_ind, v_cnt, v_dets])
        return res

    """
        gt_videos : 910个视频的标注
        gt_videos_format : 所有的tubes, 有的视频里有多个tube, 所以len(gt_videos_format) > 910
             [ [cls, video_index, np.array[[frame_index, x1, y1, x2, y2]] ] ]
               n个tubes                  
                                  n个boxes

        pred_videos_format : 每个视频,24个类别的目标框, len=910*24
            [ [cls_id, video_index, [[frame_index, np.array[[x1, y1, x2, y2, cls_score]] ]] ]
              910*24个目标框
                                    n个frame
                                                   m个box
    """
    t0 = time.time()
    gt_videos_format = gt_to_videts(gt_videos)
    t1 = time.time()
    pred_videos_format = imagebox_to_videts(all_boxes, CLASSES)
    t2 = time.time()
    print("gt_to_videts: %f sec, imagebox_to_videts %f sec" % (t1-t0, t2-t1))
    ap_all = []    
    for cls_ind, cls in enumerate(CLASSES[0:]):                         # 遍历所有class
        print("    class: ", cls)
        cls_ind += 1
        # [ video_index, [[frame_index, x1,y1,x2,y2]] ]
        gt = [g[1:] for g in gt_videos_format if g[0]==cls_ind]             # 该class的gt, [[video_index, [frame_index, x1, y1, x2, y2]]]
        pred_cls = [p[1:] for p in pred_videos_format if p[0]==cls_ind]     # 该class的目标框, [[video_index, [frame_index, [x1, y1, x2, y2, cls_score]]]]
        cls_len = None
        ap = video_ap_one_class(gt, pred_cls, iou_thresh, bTemporal, cls_len)
        ap_all.append(ap)

    return ap_all
