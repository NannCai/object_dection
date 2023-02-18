import draw_toolbox
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
cutimg_size = [448,832]

def load_pb_file(graph, pb_filename):
    with tf.gfile.GFile(pb_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')

class SSDModel():
    def __init__(self, pb_filename, size):
        self.size = size
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        load_pb_file(self.graph, pb_filename)

        self.image_input = self.graph.get_tensor_by_name("Placeholder:0")
        self.all_labels = self.graph.get_tensor_by_name("concat_3:0")
        self.all_scores = self.graph.get_tensor_by_name("concat_4:0")
        self.all_bboxes = self.graph.get_tensor_by_name("concat_5:0")

    def run(self, img, show_size = None):
        np_image = cv2.resize(img, self.size)
        np_image = np.expand_dims(np_image, 0)
        with self.graph.as_default():
            labels_, scores_, bboxes_ = self.sess.run([self.all_labels, self.all_scores, self.all_bboxes],
                feed_dict={self.image_input: np_image})

        result = []
        for i, score in enumerate(scores_):
            if score:
                bbox = list(bboxes_[i])
                # bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
                result.append([len(result)] + [labels_[i], score] + bbox)  # 0id 1label 2score bbox
        result = np.array(result)
       
        img = img.copy() if show_size is None else cv2.resize(img.copy(), show_size)
        img_to_draw = draw_toolbox.bboxes_draw_on_img(img, labels_, scores_, bboxes_, thickness=2)
        return img_to_draw, result