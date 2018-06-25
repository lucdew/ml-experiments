# Object detection with tensorflow

Here are some of my experiments with image object detection and more specifically person detection for [rpicalarm](https://github.com/lucdew/rpicalarm) project.
Most use tensorflow which is still a pain (as of June 2018) to install on raspberry pi 3 raspbian distribution for the more recent versions.

Accuracy-wise I got better results with yolov3, darknet cnn and coco dataset.

## Pre-requisites

The examples have been tested with tensorflow 1.8.X running on CPU
To install

`pip3 install tensorflow`

And the dependencies

```
pip3 install -r requirements.txt
```

## darknet (53 layers) model + yolov3 detection + coco dataset

See the article [Implmenting Yolov3 using tensorflow slim](https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe) and the associated github repository
[tensorflow-yolov3](https://github.com/mystic123/tensorflow-yolo-v3)

Download the weigthts:

```
wget https://pjreddie.com/media/files/yolov3.weights
```

Put your source images in the images folder. Result will be written in the results folder

```
cd yolov3
python3 object_detector.py
```

## darknet model + yolov2 detection

object_detector.py uses a pre-trained yolov2 model. The .pb and .meta files have been generated using [darkflow](https://github.com/thtrieu/darkflow)

Current implementation only detects the following objects:

- aeroplane
- bicycle
- bird
- boat
- bottle
- bus
- car
- cat
- chair
- cow
- diningtable
- dog
- horse
- motorbike
- person
- pottedplant
- sheep
- sofa
- train
- tvmonitor

Put your source images in the images folder. Result will be written in the results folder
