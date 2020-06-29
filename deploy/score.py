"""
Example scoring script for Keras YOLOv3 object detection
"""

import json
import numpy as np
import os
from PIL import Image
import base64
from io import BytesIO

from azureml.core.model import Model

from yolo import YOLO

def init():
    global yolo

    # Replace with your model name in the Azure ML Workspace
    model_root = Model.get_model_path('hardhat_2class_tiny_yolov3.h5', version=1)

    yolo = YOLO(model_path=os.path.join(model_root),
                anchors_path=os.path.join(os.getcwd(), 'deploy/custom_anchors.txt'),
                classes_path=os.path.join(os.getcwd(), 'deploy/custom_classes.txt'))
    
def run(raw_data):
    data = Image.open(BytesIO(base64.b64decode(json.loads(raw_data)['data'])))

    # make prediction
    r_image, bboxes, scores, classes = yolo.detect_image(data)
    return {'bboxes': bboxes.tolist(), 'scores': scores.tolist(), 'classes': classes.tolist()}