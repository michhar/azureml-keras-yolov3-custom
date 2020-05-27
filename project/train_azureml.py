"""
Retrain the YOLO v3 model for your own dataset.

Notes:
* Base model weights can be downloaded with repo links and converted
with "convert.py" to Keras format.
* The VoTT tool was used to annotate and the "voc_annotation.py" script to
convert for use by "train.py".
* There are YOLOv3 and tiny YOLOv3 options.
* Use "kmeans.py" to estimate anchors positions for the dataset and add to
an anchors file e.g. "model_data/yolo_tiny_anchors.txt"

Usage example:
    python train.py --model model_data/yolov3-tiny-weights.h5 --gpu_num 1 --annot_path example_label_list.txt --class_path model_data/yolo_obj_classes.txt --anchors_path model_data/yolo_tiny_anchors.txt
"""

import tensorflow as tf 
tf.python_io.control_flow_ops = tf

import numpy as np
import argparse
import os
import glob

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.utils import multi_gpu_model

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from yolo import YOLO

from utils import YOLO_Kmeans, convert_annotation
from converter import convert_to_keras

from azureml.core import Run, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model as AzureMLModel

# Device location (0 for 1 GPU, 0,1 for 2 GPU, and so on)
# Depends on VM SKU set for AML Compute (scale up)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dotenv import load_dotenv
from pathlib import Path  # Python 3.6+ only
env_path = Path('.') / 'myenvs'
load_dotenv(dotenv_path=env_path)


def main(gpu_num, class_path, data_dir, data_path, num_clusters):

    # Define an Azure ML run
    run = Run.get_context()

    # Folder for persistent assets
    log_dir = 'outputs/'

    # Conver to int
    num_clusters = int(num_clusters)

    # Define workspace object
    try:
        sp = ServicePrincipalAuthentication(tenant_id=os.getenv('AML_TENANT_ID'),
                                            service_principal_id=os.getenv('AML_PRINCIPAL_ID'),
                                            service_principal_password=os.getenv('AML_PRINCIPAL_PASS')) 
        ws = Workspace.get(name=os.getenv('WORKSPACE_NAME'),
                           auth=sp,
                           subscription_id=os.getenv('SUBSCRIPTION_ID'))
    # Need to create the workspace
    except Exception as err:
        print('Issue with service principal or workspace creation: ', err)
        assert False

    # Get classes
    class_names = get_classes(class_path)
    num_classes = len(class_names)

    # Create annot file (file with image path and bounding boxes - one image per line)
    image_ids = [os.path.basename(x) for x in glob.glob(os.path.join(data_path, 
        data_dir, 'JPEGImages/*.*'))]
    annot_filename = './outputs/train_val.txt'
    annot_file = open(annot_filename, 'w') # output file
    for image_id in image_ids:
        annot_file.write(os.path.join(data_path, 
            data_dir, 'JPEGImages/{}'.format(image_id)))
        convert_annotation(image_id='.'.join(image_id.split('.')[0:-1]), 
                           list_file=annot_file,
                           classes=class_names,
                           data_dir=data_dir,
                           data_path=data_path)
        annot_file.write('\n')
    annot_file.close()

    # Calculate anchor boxes
    out_file = './outputs/custom_anchors.txt'
    kmeans = YOLO_Kmeans(args.num_clusters, annot_filename, out_file)
    kmeans.txt2clusters()

    # Anchors and filters
    anchors_orig, anchors = get_anchors(out_file)
    filter_num = (num_classes + 5)*3

    # Update config file template
    # replace $FILTERS, $CLASSES, and $ANCHORS w/ correct value from above
    if num_clusters == 9:
        with open('yolov3-custom_template.cfg', 'r') as fptr:
            config_str = fptr.read()
            config_str = config_str.replace('$FILTERS', str(filter_num))
            config_str = config_str.replace('$CLASSES', str(num_classes))
            config_str = config_str.replace('$ANCHORS', anchors_orig)
    else: # num_clusters == 6
        with open('yolov3-tiny-custom_template.cfg', 'r') as fptr:
            config_str = fptr.read()
            config_str = config_str.replace('$FILTERS', str(filter_num))
            config_str = config_str.replace('$CLASSES', str(num_classes))
            config_str = config_str.replace('$ANCHORS', anchors_orig)
    with open('./outputs/yolov3-custom.cfg', 'w') as outptr:
        outptr.write(config_str)

    # Download Darknet model (from Azure ML Workspace) and convert
    keras_custom_model = './outputs/yolov3_custom.h5'
    if num_clusters == 9:
        AzureMLModel(ws, name='yolov3.weights').download(target_dir='.', exist_ok=True)
        convert_to_keras(config_path='./outputs/yolov3-custom.cfg',
                        weights_path='yolov3.weights',
                        output_path=keras_custom_model)
    else:
        AzureMLModel(ws, name='yolov3-tiny.weights').download(target_dir='.', exist_ok=True)
        convert_to_keras(config_path='./outputs/yolov3-custom.cfg',
                        weights_path='yolov3-tiny.weights',
                        output_path=keras_custom_model)


    initial_lr = 1e-2
    input_shape = (608,608) # default setting
    is_tiny_version = (num_clusters == 6)
    # Use transfer learning - all layers frozen except last "freeze_body" number
    if is_tiny_version:
        input_shape = (416,416) #Change default anchor box number
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=keras_custom_model, gpu_num=gpu_num)
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=keras_custom_model) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.2
    with open(annot_filename) as f:
        lines = f.readlines()
    
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    class LogRunMetrics(Callback):
        # callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list for Azure ML tracking
            run.log('Loss', log['loss'])

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=initial_lr), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 2
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=20,
                initial_epoch=0,
                callbacks=[logging, LogRunMetrics()])
        model.save_weights(log_dir + 'trained_weights_intermediate.h5')
        # Save architecture, too
        with open(log_dir + 'trained_weights_intermediate.json', 'w') as f:
            f.write(model.to_json())

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=initial_lr/100), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 2 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=40,
            initial_epoch=20,
            callbacks=[logging, checkpoint, early_stopping, reduce_lr, LogRunMetrics()])
        model.save_weights(log_dir + 'trained_weights_final.h5')
        # Save architecture, too
        with open(log_dir + 'trained_weights_final.json', 'w') as f:
            f.write(model.to_json())

    # Further training if needed.


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


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5', gpu_num=1):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if gpu_num>=2:
        model_body = multi_gpu_model(model_body, gpus=gpu_num)

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # Command line options

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--class_path', type=str,
        help='Text file with class names one per line.'
    )

    parser.add_argument(
        '--data_path', type=str,
        help='Azure ML path to Storage passed from datastore mount in driver.'
    )

    parser.add_argument(
        '--data_dir', type=str, default='data',
        help='Directory for training data as found in the Blob Storage container; Azure ML-specified value.'
    )

    parser.add_argument(
        '--num_clusters', type=int, default=9,
        help='Number of anchor boxes; 9 for full size YOLO and 6 for tiny YOLO.'
    )

    args = parser.parse_args()

    main(gpu_num=args.gpu_num,
         class_path=args.class_path,
         data_dir=args.data_dir,
         data_path=args.data_path,
         num_clusters=args.num_clusters)
