# -*- coding: utf-8 -*-

import numpy as np
import glob,os,sys,time
import tensorflow as tf
from PIL import Image, ImageDraw

from yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_dir', 'images', 'Input image')
tf.app.flags.DEFINE_string('output_dir', 'results', 'Output results directory')
tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_integer('size', 416, 'Image size')
tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def draw_boxes(boxes, img, cls_names, detection_size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def detect_objs(files):
    classes = load_coco_names(FLAGS.class_names)
    # placeholder for detector inputs

    start = time.time()
    saver = tf.train.import_meta_graph('yolov3-coco.meta')
    graph = tf.get_default_graph()
    #for op in graph.get_operations():
    #   print(str(op.name))
    inputs = graph.get_tensor_by_name("Placeholder:0")
    op_to_restore = graph.get_tensor_by_name("outputs:0")

    print(time.time()-start)

    with tf.Session() as sess:
      saver.restore(sess,tf.train.latest_checkpoint('./'))
      for f in files:
          start = time.time()
          img = Image.open(f)
          img_resized = img.resize(size=(FLAGS.size, FLAGS.size))
          detected_boxes = sess.run(op_to_restore,{inputs:[np.array(img_resized, dtype=np.float32)]})
          filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)

          draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))
          img.save(os.path.join(FLAGS.output_dir,os.path.basename(f)))
          print(time.time()-start)



if __name__ == '__main__':
    if not os.path.isdir(FLAGS.output_dir):
       os.makedirs(FLAGS.output_dir)

    detect_objs(glob.glob(FLAGS.input_dir+"/*.jpg"))
