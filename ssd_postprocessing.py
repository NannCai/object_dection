import cv2
import math
import struct
import numpy as np
from functools import cmp_to_key
import time

# reference code 

num_class_ = 11
softmax_on_CV22 = True
memory_alignment = True # True: sim on board False: sim on PC
offset = 0.5            # always 0.5

def get_priorbox():
    layer_shapes = [[7, 13], [14, 26], [28, 52]]
    num_anchor_scales = [3, 3, 3]
    anchor_scales = [[[0.278, 0.216], [0.375, 0.475], [0.412, 0.502]],
                    [[0.072, 0.146], [0.146, 0.108], [0.141, 0.286]],
                    [[0.024, 0.031], [0.038, 0.072], [0.079, 0.055]]]
    layer_steps = [32, 16, 8]
   
    priorbox = []

    for layer, layer_shape in enumerate(layer_shapes):
        for y in range(layer_shape[0]):
            for x in range(layer_shape[1]):
                y_on_image = (y + offset) * layer_steps[layer] / 224
                x_on_image = (x + offset) * layer_steps[layer] / 416
                for i in range(num_anchor_scales[layer]):
                    priorbox.append([y_on_image, x_on_image, anchor_scales[layer][i][0], anchor_scales[layer][i][1]])   
    return priorbox

def center2point(box):
    x_center, y_center, width, height = box
    x1 = x_center - width / 2
    x2 = x_center + width / 2
    y1 = y_center - height / 2
    y2 = y_center + height / 2
    return [x1, x2, y1, y2]

def area_box(box):
    return (box[5] - box[3]) * (box[6] - box[4])

def iou(bbox1, bbox2):
    yi1 = max(bbox1[3], bbox2[3])
    xi1 = max(bbox1[4], bbox2[4])
    yi2 = min(bbox1[5], bbox2[5])
    xi2 = min(bbox1[6], bbox2[6])
    if xi2 - xi1 <= 0 or yi2 - yi1 <= 0:
        return 0
    i = (xi2 - xi1) * (yi2 - yi1)
    u = area_box(bbox1) + area_box(bbox2) - i
    return i / u

def del_alignment(data, channel, align = 32):
    align = math.ceil(channel / align) * align
    ret_data = b""
    for i in range(len(data) // align):
        ret_data += data[i * align: i * align + channel]
    return ret_data


def import_raw_data(bin_raw, data_type, shape = None, memory_alignment = False):
    if memory_alignment:
        bin_raw = del_alignment(bin_raw, shape[-1] * 4)
    if data_type in ["float32", "float"]:
        assert len(bin_raw) % 4 == 0, "Get wrong type!"
        struct_format = "f" * (len(bin_raw) // 4)
        return np.array(struct.unpack(struct_format, bin_raw))
    else:
        print("Unknow type: ", data_type)
        quit()

def import_NCHW_data(bin_raw, data_type, shape):
    data = import_raw_data(bin_raw, "float32", shape, memory_alignment)
    if data is None:
        return None
    data = data.reshape(shape)
    if len(shape) == 4:
        data = data.transpose((0, 2, 3, 1)) # NCHW to NHWC
    elif len(shape) == 2:
        data = data.transpose((1, 0))
    else:
        print("unsupport shape")
    return data.flatten()

def ssd_postprocessing(softmax, loc):
    loc_bin_shape = [(1, 12, 7, 13), (1, 12, 14, 26), (1, 12, 28, 52)]
    priorbox = get_priorbox()

    #read conf
    conf_data = import_NCHW_data(softmax, "float32", (num_class_, 5733))
    if conf_data is None:
        return
    conf_data = conf_data.reshape([conf_data.shape[0] // num_class_, num_class_])

    # Read loc
    loc_data = np.array([])
    for i, loc_ in enumerate(loc):
        data = import_NCHW_data(loc_, "float32", loc_bin_shape[i])
        if data is None:
            return 
        loc_data = np.concatenate([loc_data, data])
    loc_data = loc_data.reshape([loc_data.shape[0] // 4, 4])
    loc_data = loc_data * np.array([0.1, 0.1, 0.2, 0.2])

    # select box
    bboxs = []

    id = 0
    for i, box in enumerate(priorbox):
        conf = np.max(conf_data[i][1:])
        class_ = conf_data[i].tolist().index(conf)
        if np.max(conf_data[i][1:]) > 0.25:
            box[0] += box[2] * loc_data[i][0]
            box[1] += box[3] * loc_data[i][1]
            box[2] *= math.exp(loc_data[i][2])
            box[3] *= math.exp(loc_data[i][3])
            x1, x2, y1, y2 = center2point(box)
            # bboxs.append([id, class_, conf, x1, y1, x2, y2])
            bboxs.append([id, class_, conf, y1, x1, y2, x2])
            id += 1

    # FIXIT ugly sort function
    for i in range(len(bboxs)):
        for j in range(i + 1, len(bboxs)):
            if bboxs[i][2] < bboxs[j][2]: # 按照置信度排列
                tmp = bboxs[i]
                bboxs[i] = bboxs[j]
                bboxs[j] = tmp
    # NMS
    delete_list = []
    for i, bbox1 in enumerate(bboxs):
        for j, bbox2 in enumerate(bboxs[i + 1:]):
            if iou(bbox1, bbox2) > 0.45:
                delete_list.append(i + j + 1)
    delete_list = list(set(delete_list))
    delete_list.sort()
    delete_list.reverse()

    for i in delete_list:
       bboxs.remove(bboxs[i])

    return bboxs