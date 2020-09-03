# YOLOv3 Keras model for general use or with Live Video Analytics

The following instructions will enable you to build a docker container with a [YOLOv3](https://arxiv.org/abs/1804.02767) [Keras](https://keras.io) custom model, built with this repository, using [nginx](https://www.nginx.com/), [gunicorn](https://gunicorn.org/), [flask](htthttps://keras.iops://github.com/pallets/flask), and [runit](http://smarden.org/runit/).  The app code is based the Keras YOLOv3 implementation for object detection at https://github.com/qqwweee/keras-yolo3.  Keras 2.2.4 and TensorFlow 1.15.2 are used in this project.  This model has not take advantage of optimizations, but may be sufficient given the device and resources in which it will run.

Note: References to third-party software in this repo are for informational and convenience purposes only. Microsoft does not endorse nor provide rights for the third-party software. For more information on third-party software please see the links provided above.

## Prerequisites
1. [Install Docker](http://docs.docker.com/docker-for-windows/install/) on your machine
2. Install [curl](http://curl.haxx.se/)
3. Custom anchors file (named `custom_anchors.txt` and placed in `app` folder)
4. Custom classes file (named `custom_classes.txt` and placed in `app` folder)
5. YOLO v3 Keras model (named `custom_model.h5` and placed in `app` folder)

## Building the docker container

1. Create a new directory on your machine and copy all the files (including the sub-folders) from this GitHub folder to that directory.
2. Build the container image (should take some minutes) by running the following docker command from a command window in that directory.

```bash
docker build . -t yolov3-keras:latest
```
    
## Running and testing

Run the container using the following docker command.

```bash
docker run --name my_yolo_container -p 80:80 -d  -i yolov3-keras:latest
```

Test the container using the following commands.

### /score

Since the LVA edge module is capable of sending specified size image in specified format, we are not preprocessing the incoming images to resize them. This is mainly because of the performance improvement.

To get a list of detected objects, use the following command.

```bash
curl -X POST http://127.0.0.1/score -H "Content-Type: image/jpeg" --data-binary @<image_file_in_jpeg>
```
If successful, you will see JSON printed on your screen that looks something like this
```json
{
  "inferences": [
    {
      "type": "entity",
      "entity": {
        "tag": {
          "value": "hardhat",
          "confidence": "0.9017146"
        },
        "box": {
          "l": "0.5199710921937251",
          "t": "0.0000000000000000",
          "w": "0.38537373941733066",
          "h": "0.8657244792833408"
        }
      }
    }
  ]
}
```

Terminate the container using the following docker commands.

```bash
docker stop my_yolo_container
docker rm my_yolo_container
```

## Upload docker image to Azure container registry

Follow instruction in [Push and Pull Docker images - Azure Container Registry](http://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli) to save your image for later use on another machine.

## Deploy as an Azure IoT Edge module

Follow instruction in [Deploy module from Azure portal](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-deploy-modules-portal) to deploy the container image as an IoT Edge module (use the IoT Edge module option). 

## Helpful links

