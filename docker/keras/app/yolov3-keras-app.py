

from timeit import default_timer as timer

import numpy as np
import cv2
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

import os
import threading
import io
import json
import copy
import time

from flask import Flask, request, jsonify, Response

class YOLO:
    _defaults = {
        "model_path": os.path.join(os.getcwd(), 'custom_model.h5'),
        "anchors_path":  os.path.join(os.getcwd(), 'custom_anchors.txt'),
        "classes_path": os.path.join(os.getcwd(), 'custom_classes.txt'),
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (608, 608),
        "gpu_num" : 0,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self._lock = threading.Lock()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self._generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        if not is_tiny_version:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), 3, num_classes)
        else:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes)         
        self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match

        assert self.yolo_model.layers[-1].output_shape[-1] == \
            num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def inference(self, cvImage):
        """Inference and postprocess"""
        detectedObjects = []
        start = timer()

        with self._lock:

            # Preprocess
            try:
                npImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
                pilImage = Image.fromarray(npImage)
                if self.model_image_size != (None, None):
                    boxed_image = letterbox_image(pilImage, tuple(reversed(self.model_image_size)))
                else:
                    new_image_size = (pilImage.width - (pilImage.width % 32),
                                    pilImage.height - (pilImage.height % 32))
                    boxed_image = letterbox_image(pilImage, new_image_size)
                image_data = np.array(boxed_image, dtype='float32')
                image_data /= 255.
                image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
            except Exception as err:
                return [{'[ERROR]': 'Error during preprocessing: {}'.format(repr(err))}]

            # Predict
            try:
                out_boxes, out_scores, out_classes = self.sess.run(
                    [self.boxes, self.scores, self.classes],
                    feed_dict={
                        self.yolo_model.input: image_data,
                        self.input_image_shape: [pilImage.size[1], pilImage.size[0]],
                        K.learning_phase(): 0
                    })

                end = timer()
                print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
                print('FPS {} for inference'.format(1./(end-start)))
            except Exception as err:
                return [{'[ERROR]': 'Error during prediciton: {}'.format(repr(err))}]

            # Postprocess
            try:
                for i, c in reversed(list(enumerate(out_classes))):
                    predicted_class = self.class_names[c]
                    box = out_boxes[i]
                    score = out_scores[i]

                    if score > self.score:
                        top, left, bottom, right = box
                        dobj = {
                                "type" : "entity",
                                "entity" : {
                                    "tag" : {
                                        "value" : predicted_class,
                                        "confidence" : str(score)
                                    },
                                    "box" : { # measured from top, left downwards
                                        "l" : str(left/float(pilImage.width)),
                                        "t" : str(top/float(pilImage.height)),
                                        "w" : str((right-left)/float(pilImage.width)), 
                                        "h" : str((bottom-top)/float(pilImage.height)) 
                                    }
                                }
                            }

                        detectedObjects.append(dobj)
            except Exception as err:
                return [{'[ERROR]': 'Error during postprocessing: {}'.format(repr(err))}]  

        return detectedObjects

    def close_session(self):
        self.sess.close()

# global ml model class
yolo = YOLO()

app = Flask(__name__)

# / routes to the default function which returns 'Hello World'
@app.route('/', methods=['GET'])
def defaultPage():
    return Response(response='Hello from Keras Yolov3 inferencing', status=200)

# /score routes to scoring function 
# This function returns a JSON object with inference duration and detected objects
@app.route('/score', methods=['POST'])
def score():
    global yolo
    try:
        # get request as byte stream
        reqBody = request.get_data(False)

        # convert from byte stream
        inMemFile = io.BytesIO(reqBody)

        # load a sample image
        inMemFile.seek(0)
        fileBytes = np.asarray(bytearray(inMemFile.read()), dtype=np.uint8)
        cvImage = cv2.imdecode(fileBytes, cv2.IMREAD_COLOR)

        # Infer Image
        detectedObjects = yolo.inference(cvImage)

        if len(detectedObjects) > 0:
            respBody = {                    
                        "inferences" : detectedObjects
                    }

            respBody = json.dumps(respBody)
            return Response(respBody, status= 200, mimetype ='application/json')
        else:
            return Response(status= 204)

    except Exception as err:
        return Response(response='[ERROR] Exception in score : {}'.format(repr(err)), status=500)

if __name__ == '__main__':
    # Run the server
    app.run(host='0.0.0.0', port=8888)