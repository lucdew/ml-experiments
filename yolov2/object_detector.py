#!/usr/bin/env python3

import json
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import os
import glob
import shutil
import urllib3
import zipfile



def expit_c(x):
    y= 1/(1+np.exp(-x))
    return y

def max_c(a,b):
    if(a>b):
        return a
    return b

def downloadModel():
    print("Downloading model")
    url = 'https://drive.google.com/uc?export=download&id=1MxEsj3XDV4j2LqOCsGbFjN4bKsKf5Gki'
    
    src_file = "tiny-yolo-voc.zip"
    urllib3.disable_warnings()
    c = urllib3.PoolManager()

    with c.request('GET',url, preload_content=False) as resp, open(src_file, 'wb') as out_file:
        shutil.copyfileobj(resp, out_file)

    resp.release_conn()

    print("Done downloading")

    with zipfile.ZipFile(src_file,"r") as zip_ref:
        zip_ref.extractall("./")

    print("Extracted zip")


class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.class_num = classes
        self.probs = np.zeros((classes,))

def nms(final_probs , final_bbox):
    boxes = list()
    class_length = final_probs.shape[1]


    input_boxes = final_bbox[:,0:4]
    input_scores = final_bbox[:,4]
    selected_indices = tf.image.non_max_suppression(
        input_boxes,
        input_scores,
        max_output_size = 5,
        iou_threshold=0.5,
        name=None
    )

    with tf.Session() as sess:
        bindices=sess.run(selected_indices)
        for idx in bindices:
            bb=BoundBox(class_length)
            bbox = final_bbox[idx]
            bb.x = bbox[0]
            bb.y = bbox[1]
            bb.w = bbox[2]
            bb.h = bbox[3]
            bb.c = bbox[4]
            bb.probs = np.asarray(final_probs[idx,:])
            boxes.append(bb)
    
    return boxes


class ObjectDetector(object):
    """
    Tensorflow image detector
    """

    def __init__(self):

        with tf.gfile.FastGFile("./tiny-yolo-voc.pb", "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(
            graph_def,
            name=""
        )

        with open("./tiny-yolo-voc.meta", 'r') as fp:
            self.meta = json.load(fp)

        self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
        self.feed = dict() # other placeholders
        self.out = tf.get_default_graph().get_tensor_by_name('output:0')

    def resize_input(self, im):
        h, w, c = self.meta['inp_size']
        imsz = cv2.resize(im, (w, h))
        imsz = imsz / 255.
        imsz = imsz[:,:,::-1]
        return imsz

    def load_image(self,image_path):
        return  cv2.imread(image_path)

    def find_boxes(self,net_out_in):
        H, W, _ = self.meta['out_size']
        C = self.meta['classes']
        B = self.meta['num']
        threshold = self.meta['thresh']

        anchors = np.asarray(self.meta['anchors'])
        tempc=0.0
        
        net_out = net_out_in.reshape([H, W, B, int(net_out_in.shape[2]/B)])
        classes = net_out[:, :, :, 5:]
        box_pred = net_out[:, :, :, :5]
        probs = np.zeros((H, W, B, C), dtype='f')
        for row in range(H):
            for col in range(W):
                for box_loop in range(B):
                    asum=0
                    arr_max=0

                    box_pred[row, col, box_loop, 4] = expit_c(box_pred[row, col, box_loop, 4])
                    box_pred[row, col, box_loop, 0] = (col + expit_c(box_pred[row, col, box_loop, 0])) / W
                    box_pred[row, col, box_loop, 1] = (row + expit_c(box_pred[row, col, box_loop, 1])) / H
                    box_pred[row, col, box_loop, 2] = np.exp(box_pred[row, col, box_loop, 2]) * anchors[2 * box_loop + 0] / W
                    box_pred[row, col, box_loop, 3] = np.exp(box_pred[row, col, box_loop, 3]) * anchors[2 * box_loop + 1] / H
                
                    # Softmax
                    for class_loop in range(C):
                        arr_max=max_c(arr_max,classes[row,col,box_loop,class_loop])
                                        
                    for class_loop in range(C):
                        classes[row,col,box_loop,class_loop]=np.exp(classes[row,col,box_loop,class_loop]-arr_max)
                        asum+=classes[row,col,box_loop,class_loop]
                    
                    for class_loop in range(C):
                        tempc = classes[row, col, box_loop, class_loop] * box_pred[row, col, box_loop, 4]/asum
                        if(tempc > threshold):
                            #print("got a match")
                            probs[row, col, box_loop, class_loop] = tempc
                            #print(tempc)

        return nms(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(box_pred).reshape(H*B*W,5))


    def detect_objects(self,image_path,outfolder=".",prefix="result_"):
        imgcv=self.load_image(image_path)
        imz = self.resize_input(imgcv)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        this_inp = np.expand_dims(imz, 0)
        feed_dict = {self.inp: this_inp}
        net_out = sess.run(self.out, feed_dict)
        boxes = self.find_boxes(net_out[0])
        self.draw_boxes(boxes,imgcv)

        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)

        img_name = os.path.join(outfolder, prefix+os.path.basename(image_path))
        print(img_name)
        cv2.imwrite(img_name, imgcv)



    def process_box(self, b, h, w, threshold):
        max_indx = np.argmax(b.probs)
        max_prob = b.probs[max_indx]
        label = self.meta['labels'][max_indx]
        if max_prob > threshold:
            left  = int ((b.x - b.w/2.) * w)
            right = int ((b.x + b.w/2.) * w)
            top   = int ((b.y - b.h/2.) * h)
            bot   = int ((b.y + b.h/2.) * h)
            if left  < 0    :  left = 0
            if right > w - 1: right = w - 1
            if top   < 0    :   top = 0
            if bot   > h - 1:   bot = h - 1
            mess = '{}'.format(label)
            return (left, right, top, bot, mess, max_indx, max_prob)
        return None

    def draw_boxes(self,boxes,imgcv):
        meta = self.meta
        h, w, _ = imgcv.shape
        threshold = meta['thresh']
        colors = meta['colors']

        for box in boxes:
            boxResults = self.process_box(box, h, w, threshold)
            if boxResults is None:
                continue
            
            left, right, top, bot, mess, max_indx, _ = boxResults
            thick = int((h + w) // 300)
            
            cv2.rectangle(imgcv, (left, top), (right, bot), colors[max_indx], thick)
            cv2.putText(imgcv, mess, (left, top - 12),0, 1e-3 * h, colors[max_indx], thick // 3)



if __name__ == "__main__":
    
    # Print labels
    #for label in OBJ_DETECTOR.meta['labels']:
    #    print(label)
    if not os.path.isfile("./tiny-yolo-voc.meta") or not os.path.isfile("./tiny-yolo-voc.pb"):
        downloadModel()

    OBJ_DETECTOR = ObjectDetector()

    for f in glob.glob("images/*.jpg"):
        print("Processing %s"%f)
        t = time.process_time()    
        OBJ_DETECTOR.detect_objects(f,outfolder="results",prefix="")
        elapsed_time = time.process_time() - t
        print("Elapsed time %s"%str(elapsed_time))
