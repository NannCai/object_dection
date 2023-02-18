import draw_toolbox
import cv2
from ssd_model import SSDModel
import numpy as np
import os
import collections

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
cutimg_size = [448,832] # y x

def img_cut(cut):
    return [cut[0],cutimg_size[0]+cut[0],cut[1],cut[1]+cutimg_size[1]]  # y y x x

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

def iou2(bb, bbgt):
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
    iw = bi[2] - bi[0]
    ih = bi[3] - bi[1]
    a1= (bb[2] - bb[0]) * (bb[3] - bb[1])
    a2= (bbgt[2] - bbgt[0]) * (bbgt[3] - bbgt[1])
    if iw > 0 and ih > 0:
        ov1 = iw * ih / ((bb[2] - bb[0]) * (bb[3] - bb[1]))
        ov2 = iw * ih / ((bbgt[2] - bbgt[0]) * (bbgt[3] - bbgt[1]))
        return max(ov1, ov2)
    return -1

def class_select(result):
    result_= []
    for bbox in result:
        if bbox[1] in [1, 2, 3]: # car truck bus
            bbox = bbox.tolist()  # label append
            bbox.append(-1)
            result_.append(bbox)
    result_ = np.array(result_)
    return result_

def crop2oirg_size(img, crop_result):
    for crop_bbox in crop_result:
        crop_bbox[3:7] = [(crop_bbox[i + 3] * cutimg_size[i % 2] + cut[i % 2])/img.shape[i % 2] for i in range(4)]
    return crop_result

def id_search(result, id):
    for orig_bbox in result:
        if orig_bbox[0] == id:
            return orig_bbox

def iou_match(orig_result, crop_result, MINOVERLAP = 0.5):   # iou>0.5 match
    match_dict = {}
    crop_no_match = []
    crop_result.tolist()
    crop_result_ = crop_result
    crop_result = collections.deque(crop_result)  # 就是一个动态栈（要用的再放回来）
    while crop_result:
        ovmax = -1
        crop_bbox = crop_result.popleft().tolist()
        is_None_true = True
        for orig_bbox in orig_result:  #有记录功能
            ov = iou(orig_bbox[3:7], crop_bbox[3:7])   # 1v4的ov算错了
            if ov > ovmax and ov > MINOVERLAP: 
                if match_dict.get(crop_bbox[0], None) == None: # 1orig
                    if orig_bbox[7] == -1:  # 1crop 1orig
                        match_dict[crop_bbox[0]] = orig_bbox[0]
                        orig_bbox[7] = crop_bbox[0]
                        is_None_true = False
                        ovmax = ov
                    else:  # 2crop 1orig
                        old_crop_id = orig_bbox[7]
                        old_crop = id_search(crop_result_, old_crop_id)
                        new_crop = crop_bbox                   
                        old_ov = iou(orig_bbox[3:7], old_crop[3:7])
                        new_ov = iou(orig_bbox[3:7], new_crop[3:7])
                        if old_ov < new_ov:
                            match_dict.pop(old_crop[0])
                            crop_result.append(old_crop)
                            match_dict[new_crop[0]] = orig_bbox[0]
                            orig_bbox[7] = new_crop[0]
                            is_None_true = False
                            ovmax = ov

                else: # 2orig
                    old_orig_id = int(match_dict[crop_bbox[0]])
                    old_orig = id_search(orig_result, old_orig_id)
                    new_orig = orig_bbox
                    old_ov = iou(old_orig[3:7], crop_bbox[3:7])
                    new_ov = iou(new_orig[3:7], crop_bbox[3:7])
                    if old_ov < new_ov:   #更大也不一定用
                        old_crop = id_search(crop_result_, old_orig[7])
                        old_ov = iou(new_orig[3:7], old_crop[3:7])
                        new_ov = iou(new_orig[3:7], crop_bbox[3:7])
                        if new_ov > old_ov: # 更大一定用
                            match_dict.pop(old_crop[0])
                            crop_result.append(old_crop)     # 被覆盖的key放到队尾
                            match_dict[crop_bbox[0]] = new_orig[0]  # dict
                            orig_bbox[7] = new_crop[0]
                    is_None_true = False
        if is_None_true == True:
            crop_no_match.append(crop_bbox)
    return match_dict, crop_no_match

def cut_fusion(bbox, cut, abstand = 20): # TODO abstand要调、尺寸要传进来
    if bbox[1] > cut[1]/1920 and bbox[1] < (cut[1] + abstand)/1920:# left edge
        return "left"
    if bbox[3] > (cut[1] + cutimg_size[1] - abstand)/1920 and bbox[3] < (cut[1] + cutimg_size[1])/1920: # right edge
        return "right"
    return False

def edge_match(crop_no_match, orig_no_match, cut, MINOVERLAP = 0.8):  # TODO up and down bbox edge
    mix_result = []
    no_match = []
    for crop_bbox in crop_no_match:
        for orig_bbox in orig_no_match:
            ov = iou2(crop_bbox[3:7],orig_bbox[3:7])
            if ov > MINOVERLAP :
                crop_bbox[7] = orig_bbox[7] = 1
                edge = cut_fusion(crop_bbox[3:7], cut) # 各保留两个坐标
                mix_bbox = orig_bbox
                if orig_bbox[1] in [2, 3] or crop_bbox[2] < 0.3:
                    mix_bbox[1]
                if edge == "left":  # car (bus truck 另外)
                    mix_bbox[3] = crop_bbox[3]
                    mix_bbox[6] = crop_bbox[6]
                    mix_result.append(mix_bbox)
                elif edge == "right":
                    mix_bbox = orig_bbox
                    mix_bbox[3] = crop_bbox[3]
                    mix_bbox[4] = crop_bbox[4]
                    mix_result.append(mix_bbox)
        if crop_bbox[7] == -1:
            no_match.append(crop_bbox)

    for bbox in orig_no_match:
        if bbox[7] == -1:
            no_match.append(bbox)
       
    return mix_result, no_match

def match_mix(match_dict, orig_result, crop_result, cut, abstand = 10):  # TODO k的取值
    match_result = []
    for crop_id in match_dict.keys():
        crop_bbox = id_search(crop_result, crop_id)
        orig_bbox = id_search(orig_result, match_dict[crop_id])
        mix_bbox = crop_bbox
        if crop_bbox[4] >= cut[1]/1920 and crop_bbox[4] <= (cut[1] + abstand)/1920:   # left edge 等号必须要
            mix_bbox[4] = orig_bbox[4]
        elif crop_bbox[6] >= (cut[1] + cutimg_size[1] - abstand)/1920 \
            and crop_bbox[6] <= (cut[1] + cutimg_size[1])/1920:  # right edge
            mix_bbox[6] = orig_bbox[6]
        elif orig_bbox[2] * 0.4 > crop_bbox[2]:  # TODO not edge sore * k
            mix_bbox = orig_bbox
        # else:
        #     mix_bbox = crop_bbox
        match_result.append(mix_bbox)

   
    return match_result

def one_pic_mix(abs_path, cut):
    img = cv2.imread(abs_path)
    img1 = img.copy()
    img_edge = img_cut(cut) # x1 x2 y1 y2
    img2 = img1.copy()[img_edge[0]:img_edge[1],img_edge[2]:img_edge[3]]  # y y x x

    # get bbox
    crop_show, crop_result = ssd_big.run(img2)
    orig_show, orig_result = ssd_small.run(img)

    # select only car truck
    crop_result = class_select(crop_result)
    orig_result = class_select(orig_result)

    # crop2oirg
    crop_result = crop2oirg_size(img, crop_result)

    match_dict, crop_no_match = iou_match(orig_result, crop_result)  
    match_result = match_mix(match_dict, orig_result, crop_result, cut)

    orig_no_match = []
    for ori_bbox in orig_result:
        if ori_bbox[7] == -1:
            orig_no_match.append(ori_bbox)
    mix_result, no_match = edge_match(crop_no_match, orig_no_match, cut)

   
    # result = match_result + mix_result + no_match
    result = crop_result.tolist()+orig_result.tolist()

    if len(result) != 0:
        result = np.array(result)
        to_show = draw_toolbox.bboxes_draw_on_img(img, result[:,1].astype(dtype=int),result[:,2],result[:,3:7], thickness=1)
    else:
        to_show = img
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(to_show, abs_path.split('/')[-1],(1650,25), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    to_show = cv2.resize(to_show,None,fx=0.7,fy=0.7,interpolation=cv2.INTER_CUBIC)
    return to_show


if __name__ == "__main__":
    ssd_small = SSDModel("D:/bbox_model/ssd_small.pb",(352,160))
    ssd_big = SSDModel("D:/bbox_model/ssd_big.pb",(416,224))  # 网络尺寸更大
    cut = [300,520] # top left

    path = "D:/bbox_model/diff_img/LM-DS2020-2-04593.jpg"
    to_show = one_pic_mix(path, cut)

    cv2.imshow("img", to_show)
    if cv2.waitKey(0) == ord("q"):
        quit()


    # pic_file = 'D:/bbox_model/diff_img'
    # for img_name in os.listdir(pic_file):
    #     # print(pic_file + "/" + img_name) # D:/data/100degee/LM-DS2020-2/LM-DS2020-2_04000-05999/LM-DS2020-2_4000-5999_pic/LM-DS2020-2-04000.jpg

    #     abs_path = pic_file + "/" + img_name
    #     to_show = one_pic_mix(abs_path, cut)

    #     print('D:/bbox_model/deque_img/' + img_name)
    #     cv2.imwrite('D:/bbox_model/deque_img/' + img_name, to_show)