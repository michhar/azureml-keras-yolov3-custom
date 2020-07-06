"""
ONNX inference test to ensure model loads and performs inference.
Prints bounding boxes, classes and saves results as image overlay.
"""
from keras.preprocessing import image
from keras import backend as K
import tensorflow as tf
import onnx
import onnxruntime
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import colorsys
from timeit import default_timer as timer
import argparse


def main(args):
    """
    Discovers bounding boxes and class.  Overlays boxes
    on original image and saves to same folder.

    Arguments
    ---------
    args : dict
        Contents:
        args.image - image path on disk
        args.model_local - model path on disk
        args.classes_path - classes file on disk
        args.anchors_path - anchors file on disk
        args.conf - confidence score threshold
    """
    # Class names and anchors from file paths
    class_names = get_class(args.classes_path)
    anchors = get_anchors(args.anchors_path)
    
    if len(anchors) == 6:
        model_size = 416 # tiny
    elif len(anchors) == 9:
        model_size = 608
    else:
        print('Number of anchors is incorrect.  Please check the anchor file.')
        assert False

    # Image preprocessing
    img_path = args.image   # TODO:  make sure the image is in img_path
    image = Image.open(img_path)
    orig_w, orig_h = image.size # PIL is (width, height)
    img = image.resize((model_size, model_size))
    x = np.array(img, dtype='float32')
    x /= 255.
    x = np.expand_dims(x, 0)

    # Check model
    model = onnx.load(args.model_local)
    onnx.checker.check_model(model)
    ## To print graph of ONNX model, uncomment the following line
    # print(onnx.helper.printable_graph(model.graph))
    
    # Runtime prediction
    sess = onnxruntime.InferenceSession(args.model_local)
    x = x if isinstance(x, list) else [x]
    feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
    start = timer()
    outputs = sess.run(None, feed)

    # Post-process ONNX model output (output conv layers)
    out_boxes, out_scores, out_classes = yolo_eval(outputs, 
                anchors=anchors,
                num_classes=len(class_names), 
                image_shape=[model_size, model_size],
                score_threshold=args.score,
                iou_threshold=0.3)

    # Tensor to float and print
    out_boxes = K.eval(out_boxes)
    out_scores = K.eval(out_scores)
    out_classes = K.eval(out_classes)

    end = timer()
    print('{} seconds for inference with ONNX runtime'.format((end-start)))

    # Resize to original dimensions
    out_boxes[:,[0,2]] *= orig_h // model_size
    out_boxes[:,[1,3]] *= orig_w // model_size

    # Print
    for i, c in enumerate(out_classes):
        score = out_scores[i]
        box = out_boxes[i]
        print(box, ' ', class_names[c])

    # Formatting
    thickness = (image.size[0] + image.size[1]) // 300
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                    for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        if score > args.score:
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

    # Save image w/ bbox
    suffix = '.'+img_path.split('.')[-1]
    save_path = image.save(img_path.replace(suffix, '_onnx_out.'+suffix))
    image.show()

def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    feats = K.constant(feats)
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """Process Conv layer output"""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-local', type=str, dest='model_local',
        help='The path to the local ONNX model.'
    )
    parser.add_argument(
        '--image', type=str, dest='image',
        help='The path to a test image.'
    )
    parser.add_argument(
        '--anchors-path', type=str, dest='anchors_path',
        help='File path to anchor definitions'
    )
    parser.add_argument(
        '--classes-path', type=str, dest='classes_path',
        help='File path to class definitions'
    )
    parser.add_argument(
        '--conf', type=float, default=0.3, dest='score',
        help='[Optional] Confidence score'
    )
    args = parser.parse_args()
    main(args)