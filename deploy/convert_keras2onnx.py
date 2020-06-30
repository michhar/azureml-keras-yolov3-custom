"""
Converts the yolo model to ONNX format so that you can take advantage of the ONNX
runtime.

Must run from this directory, "deploy", to utlize the yolo class files.

Necessary libraries and versions to make this work:
tensorflow==1.15.2
keras==2.2.4
keras2onnx==1.7.0
onnx==1.6.0
"""
import onnx
import keras2onnx
import argparse
import os
import numpy as np

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model

from yolo import YOLO
from yolo3.model import yolo_body, tiny_yolo_body


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors_orig = f.readline()
    anchors = [float(x) for x in anchors_orig.split(',')]
    return anchors_orig, np.array(anchors).reshape(-1, 2)

def main(args):

    # Get classes
    class_names = get_classes(args.class_path)
    num_classes = len(class_names)

    # Anchors
    anchors_orig, anchors = get_anchors(args.anchor_path)
    num_anchors = len(anchors)

    # input_shape = (608,608) # default setting
    is_tiny_version = (args.num_clusters == 6)

    # Load model, or construct model and load weights.
    if not is_tiny_version:
        yolo_model = yolo_body(Input(shape=(None, None, 3)), 3, num_classes)
    else:
        yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes)         
    yolo_model.load_weights(args.model_local) # make sure model, anchors and classes match

    assert yolo_model.layers[-1].output_shape[-1] == \
        num_anchors/len(yolo_model.output) * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes'


    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(yolo_model, 'yolov3')

    # Save as ONNX format
    temp_model_file = args.name
    # keras2onnx.save_model(onnx_model, temp_model_file)
    onnx.save_model(onnx_model, temp_model_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-local', type=str, dest='model_local',
        help='The path to the local model.'
    )
    parser.add_argument(
        '--class-path', type=str, dest='class_path',
        help='Text file with class names one per line.'
    )
    parser.add_argument(
        '--anchor-path', type=str, dest='anchor_path',
        help='Text file with anchor box sizes.'
    )
    parser.add_argument(
        '--num-clusters', type=int, dest='num_clusters',
        help='6 for tiny and 9 for full YOLO v3 models.'
    )
    parser.add_argument(
        '--name', type=str, dest='name',
        help='Exported model name (common to use extension .onnx)'
    )

    args = parser.parse_args()
    main(args)