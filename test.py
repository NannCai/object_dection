import draw_toolbox
import cv2
from ssd_model import SSDModel
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
cutimg_size = [448,832]
# score > 0.995 (value need check)
# cut_edge ? to score

# in order to select pictures that can be shown

def img_cut(cut):
    return [cut[0],cutimg_size[0]+cut[0],cut[1],cut[1]+cutimg_size[1]]

def iou(bb, bbgt):
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
    iw = bi[2] - bi[0]
    ih = bi[3] - bi[1]
    if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        ua = (bb[2] - bb[0]) * (bb[3] - bb[1]) + (bbgt[2] - bbgt[0]
            ) * (bbgt[3] - bbgt[1]) - iw * ih
        return  iw * ih / ua
    return -1

def class_select(result):
    result_= []
    for bbox in result:
        if bbox[1] in [1, 2, 3]: # car truck bus
            result_.append(bbox)
    result_ = np.array(result_)
    return result_

def crop2oirg_size(img, crop_result):
    for crop_bbox in crop_result:
        crop_bbox[3:7] = [(crop_bbox[i + 3] * cutimg_size[i % 2] \
            + cut[i % 2])/img.shape[i % 2] for i in range(4)]
    return crop_result

def mix_min_area(orig_result, crop_result, MINOVERLAP = 0.5):
    MINOVERLAP = 0.5
    result = []
    for ori_bbox in orig_result:
        ovmax = -1
        for crop_bbox in crop_result:
            ov = iou(ori_bbox[3:7], crop_bbox[3:7])
            if ov > ovmax:
                ovmax = ov
                match_det = crop_bbox

        if ovmax >= MINOVERLAP:  # 找到匹配精细框
            ori_area = (ori_bbox[5] - ori_bbox[3]) * (ori_bbox[6] - ori_bbox[4])
            crop_area = (match_det[5] - match_det[3]) * (match_det[6] - match_det[4])
            if ori_area >= crop_area:
                result.append(match_det)
            else:
                result.append(ori_bbox)
        else:  # 精细框范围外 则取原始框
            result.append(ori_bbox)

    for crop_bbox in crop_result:
        ovmax = -1
        for ori_bbox in orig_result:
            ov = iou(ori_bbox[3:7], crop_bbox[3:7])
            if ov > ovmax:
                ovmax = ov
        if ovmax < MINOVERLAP:
            result.append(crop_bbox)
    return result

def score_choose(orig_result, crop_result, MINOVERLAP = 0.5): # 根据置信度选择
    MINOVERLAP = 0.5
    result = []
    for ori_bbox in orig_result:
        ovmax = -1
        for crop_bbox in crop_result:
            ov = iou(ori_bbox[3:7], crop_bbox[3:7])
            if ov > ovmax:
                ovmax = ov
                match_det = crop_bbox
   
        if ovmax >= MINOVERLAP:  # 找到匹配精细框4 todo

            if match_det[2] >= ori_bbox[2]:
                result.append(match_det)
            else:
                result.append(ori_bbox)

        else:  # 精细框范围外 则取原始框
            result.append(ori_bbox)

    for crop_bbox in crop_result:
        ovmax = -1
        for ori_bbox in orig_result:
            ov = iou(ori_bbox[3:7], crop_bbox[3:7])
            if ov > ovmax:
                ovmax = ov
        if ovmax < MINOVERLAP:
            result.append(crop_bbox)
    return result

def one_pic_mix(abs_path):
    img = cv2.imread(abs_path)
    img1 = img.copy()
    img_edge = img_cut(cut)
    img2 = img1.copy()[img_edge[0]:img_edge[1],img_edge[2]:img_edge[3]]

    # get bbox
    crop_show, crop_result = ssd_big.run(img2)
    orig_show, orig_result = ssd_small.run(img)

    # select only car truck
    crop_result = class_select(crop_result)
    orig_result = class_select(orig_result)

    # crop2oirg
    crop_result = crop2oirg_size(img, crop_result)
    result = score_choose(orig_result, crop_result)

    if len(result) != 0:
        result = np.array(result)
        to_show = draw_toolbox.bboxes_draw_on_img(img, \
            result[:,1].astype(dtype=int),result[:,2],result[:,3:], thickness=1)
    else:
        to_show = img
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(to_show, abs_path.split('/')[-1],(1650,25), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    return to_show


if __name__ == "__main__":
    ssd_small = SSDModel("D:/bbox_model/ssd_small.pb",(352,160))
    ssd_big = SSDModel("D:/bbox_model/ssd_big.pb",(416,224))  # 网络尺寸更大
    cut = [300,520] # top left
    img_edge = img_cut(cut)


    pic_file = 'D:/data/100degee/LM-DS2020-2/LM-DS2020-2_04000-05999/LM-DS2020-2_4000-5999_pic'
    for img_name in os.listdir(pic_file):
        # print(pic_file + "/" + img_name) # D:/data/100degee/LM-DS2020-2/LM-DS2020-2_04000-05999/LM-DS2020-2_4000-5999_pic/LM-DS2020-2-04000.jpg

        abs_path = pic_file + "/" + img_name
        to_show = one_pic_mix(abs_path)
        # print('D:/bbox_model/test_img/' + img_name)
        # quit()
        cv2.imwrite('D:/bbox_model/test_img/score_choose_' + img_name, to_show)