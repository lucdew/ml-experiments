# Object detection with tensorflow and yolov2 model

Here are some of my experiments with image object detection and more specifically person detection for [rpicalarm](https://github.com/lucdew/rpicalarm) project.
Most use tensorflow which is still a pain (as of June 2018) to install on raspberry pi 3 raspbian distribution for the more recent versions.

## Using pre-trained model

object_detector.py uses a pre-trained yolov2 model. The .pb and .meta files have been generated using [darkflow](https://github.com/thtrieu/darkflow).

Current implementation only detects the following objects:
* aeroplane
* bicycle
* bird
* boat
* bottle
* bus
* car
* cat
* chair
* cow
* diningtable
* dog
* horse
* motorbike
* person
* pottedplant
* sheep
* sofa
* train
* tvmonitor


Put your source images in the images folder. Result will be written in the results folder