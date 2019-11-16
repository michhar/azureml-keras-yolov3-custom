# A Keras Implementation of YOLO v3 for Custom Model Training with Azure Machine Learning

Keras is a deep learning framework that operates as a binding to lower level frameworks such as TensorFlow and CNTK.  Azure Machine Learning, an ML platform integrated with Microsft Azure for data prep, experimentation and model deployment, is exposed through a Python SDK (used here) and extension to the Azure CLI.  Together a custom Keras model for object detection is trained using the code and instruction in theis repo.  The ML practitioner must bring their own custom data to this process - hence any object detector can be trained by following the process below.

This work is based on:

* Keras YOLOv3 implementation for object detection https://github.com/qqwweee/keras-yolo3
* A fork for custom data https://github.com/michhar/keras-yolo3-custom (this repo is the Azure ML implementation).

YOLO stands for you only look once and is an efficient algorithm for object detection.  Here, for example, a YOLOv3 detector was trained to detect large trucks.

<img src="assets/truck_id.png" width="75%" alignment="center">

Important papers on YOLO:

* Original - https://arxiv.org/abs/1506.02640
* 9000/v2 - https://arxiv.org/abs/1612.08242
* v3 - https://arxiv.org/abs/1804.02767

There are "tiny" versions of the architecture, often considered for embedded/constrained devices.

Website:  https://pjreddie.com/darknet/yolo/ (provides information on a framework called Darknet)

This implementation of YOLOv3 (Tensorflow backend) was inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).

---

## What You Do Here

Using this repo you will perform some or all of the following:

* Convert a Darknet model to a Keras model (and if custom setup, modify the config file with proper filter and class numbers)
* Perform inference on video or image - a quick start
* Label data with a bounding box definition tool
* Train a model with Azure Machine Learning on custom data using the converted custom Darknet model in Keras format (`.h5`)
* Perform inference with the custom model on single image frames or videos

## Quick Start Demo on Local Computer

### Set up your environment for demo

* Python 3 is recommended (tested with an Anaconda install of 3.6)

1. `pip install -r requirements.txt`

### Demo

2. Download YOLOv3 weights.

Download full-sized YOLOv3 here:  https://pjreddie.com/media/files/yolov3.weights

Or, on linux:  `wget https://pjreddie.com/media/files/yolov3.weights`

  * If the tiny version of the weights are needed, download this file:  https://pjreddie.com/media/files/yolov3-tiny.weights

3. Convert the Darknet YOLOv3 model to a Keras model.

  * To convert the darknet format of weights to Keras format, make sure you have run the following using the proper config file

      `python convert.py -w cfg/yolov3.cfg yolov3.weights yolo_weights.h5`

4. Run YOLO detection with `yolo_video.py`.


Run on video or image file:

```
usage: yolo_video.py [-h] [--model_path MODEL_PATH]
                     [--anchors_path ANCHORS_PATH]
                     [--classes_path CLASSES_PATH] [--gpu_num GPU_NUM]
                     [--image] [--input [INPUT]] [--output [OUTPUT]]

  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to model weight file, default model_data/yolo.h5
  --anchors_path ANCHORS_PATH
                        path to anchor definitions, default
                        model_data/yolo_anchors.txt
  --classes_path CLASSES_PATH
                        path to class definitions, default
                        model_data/coco_classes.txt
  --gpu_num GPU_NUM     Number of GPU to use, default 1
  --image               Image detection mode, will ignore all positional
                        arguments
  --input [INPUT]       Video input path
  --output [OUTPUT]     [Optional] Video output path
```

For examples, see below in [Use model](#use-model).

e.g.  `python yolo_video.py --model_path <path to converted model>/yolo_weights.h5 --anchors project/yolo_anchors.txt --classes_path project/coco_classes.txt`

> For Tiny YOLOv3, just do in a similar way, except with tiny YOLOv3, converted weights.

---

4. MultiGPU usage is an optional. Change the number of gpu and add gpu device id.

## Data Prep

1. Use the VoTT **v1** (<a href="https://github.com/microsoft/VoTT/releases/tag/v1.7.2">link to download</a>) labeling tool if using custom data and export to **Tensorflow Pascal VOC**.

You should get some name of a folder followed by `_output`.  It will have three subfolders
```
/Annotations
/ImageSets
/JPEGImages
```

2. Rename the `xyz_output` folder to `data`.

3. Copy `data` to the base of this repo.

## Labels

1. Generate your own annotation file and class names file in the following format with `voc_annotation.py` described below.

    * It should generate one row for each image, e.g.:  `image_file_path x_min,y_min,x_max,y_max,class_id`.

Generate this file with `voc_annotation.py`.  This script takes all of the `Annotations` and `ImageSets` (classes with val and train) and converts them into one file.  The final text file will then be used by the training script.

  * Modify the `sets` and `classes` appropriately within `voc_annotation.py`
  * Run with `python voc_annotation.py`

    Here is an example of the output:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

## Other Project Settings

2.  Calculate the appropriate achor box sizes with `kmeans.py` using the training annotation file output from `voc_annotation.py` as the `--annot_file`.

    `python kmeans.py --annot_file project/train_list.txt --out_file project/custom_anchors.txt --num_clusters 9`

    * Note, for  `--num_clusters`, use 9 for full YOLOv3 and 6 for tiny YOLOv3.

3. To convert the darknet format of weights to Keras format, make sure you have run the following using the proper config file (with, optionally, the custom achors from the previous step)

IMPORTANT NOTES ON CONFIG:

  * Make sure you have set up the config `.cfg` file correctly (`filters` and `classes`) - more information on how to do this <a href="https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects" target="_blank">"How to train (to detect your custom objects):" (do only step #1)</a>
  * For config, update the `filters` in CNN layer above `[yolo]`s and `classes` in `[yolo]`'s to class number (as well as anchors if using custom anchors!)


    `python convert.py -w project/yolov3-custom1class.cfg yolov3.weights project/yolov3-custom-base.h5`  

  * The file `yolov3-custom-base.h5` is, next in training, used to load pretrained weights.

## Data Storage

### Create Storage

1. In Azure, create Storage account

### Set Environment Variables

**Windows**

Create a `setenvs.cmd` file with the following:

```
set STORAGE_CONTAINER_NAME_TRAINDATA=<Blob container name for trained data>
set STORAGE_ACCOUNT_NAME=<Storage account name>
set STORAGE_ACCOUNT_KEY=<Storage account key>
```

**Linux/MacOS**

Create a `setenvs.sh` file with the following:

```
export STORAGE_CONTAINER_NAME_TRAINDATA=<Blob container name for trained data>
export STORAGE_ACCOUNT_NAME=<Storage account name>
export STORAGE_ACCOUNT_KEY=<Storage account key>
```

Run the .cmd or shell script to set the environment variables in the current terminal window.

### Upload Data to Azure Blob

Note:  just need Azure Blob Storage account (a new container is created).

Run:

    python upload_to_blob.py --container <container name> --account <storage account name> --key <storage account key> --dir data

## Train with Azure ML

Here, a driver script will be used to control the training with Azure ML on an Azure ML Compute cluster, provisioned or set in driver script, along with the data store and config files to use.  Some settings will need to be mofified.

1.  On command line log in to Azure with the Azure CLI and ensure in correct subscription (may need to `az account set --subscription <subscription name>`)

2. Need to have `./project/.azureml/config.json` which specified Azure ML Workspace.

3. Need to update `epochs` and `initial_epoch` in `train_azureml.py` in project folder.

[Add guidance here]

4. Modify `azureml_driver.py` with correct settings.  Run the following (no arguments so please set in script as needed).

    python azureml_driver.py

## Use Model

Download from Azure ML Workspace and the correct Experiment run outputs.

### Inference on an image

In addition to other arguments, use `--image`

Example:  `python yolo_video.py --model_path trained_weights_final.h5 --anchors project/custom_anchors.txt --classes_path project/custom_classes.txt --image`

### Inference on video from a webcam

Note:  on linux `video0` is usually the built-in camera (if this exists) and a USB camera may be used so look for `video1` etc.  (if there is not camera, then `video0` is usually USB cam).  On MacOS, use for `--input` 0, for built-in, and 1 for USB.  This should be the same on Windows.

In addition to other arguments, use `--input <video device id>`

Example:  `python yolo_video.py --model_path trained_weights_final.h5 --anchors project/custom_anchors.txt --classes_path project/custom_classes.txt --input 0`

### Inference on video file and output to a video file

In addition to other arguments, use `--input <video file name> --output xyz.mov`

Example:  `python yolo_video.py --model_path trained_weights_final.h5 --anchors project/custom_anchors.txt --classes_path project/custom_classes.txt --input <path to video>/some_street_traffic.mov --output some_street_traffic_with_bboxes.mov`

## Credits

* Based on https://github.com/qqwweee/keras-yolo3


## References

* [Azure Machine Learning documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
* [Building Powerful Image Classfication Models Using ery Little Data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) 
---

## Some issues to know

1. Default anchors can be used. If you use your own anchors, probably some changes are needed (using `model_data/yolo_tiny_anchors.txt`).

2. The inference result is not totally the same as Darknet but the difference is small.

3. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

4. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

5. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.