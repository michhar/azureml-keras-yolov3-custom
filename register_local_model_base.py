"""
Register a local model to Azure ML workspace
"""

import azureml
from azureml.core import Workspace
from azureml.core.model import Model

import os
import argparse


def main(args):

    # Used downloaded Workspace configuratin file to authenticate
    # interactively
    try:    
        ws = Workspace.from_config(path='./project/.azureml/config.json')
    # Need to create the workspace
    except Exception as err:
        print('No workspace.  Check for config.json file under "project/.azureml".')
        assert False

    if args.model_size == 'full':
        model = Model.register(model_path="yolov3.weights",
                            model_name="yolov3.weights",
                            tags={'framework': "darknet", 'type': "object detection"},
                            description="Darknet base model trained on COCO (80 classes)",
                            workspace=ws)
    elif args.model_size == 'tiny':
        model = Model.register(model_path="yolov3-tiny.weights",
                            model_name="yolov3-tiny.weights",
                            tags={'framework': "darknet", 'type': "object detection"},
                            description="Darknet base model (tiny version) trained on COCO (80 classes)",
                            workspace=ws)
    else:
        print('Please choose "full" or "tiny" for your "--model-size" argument.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-size', type=str, dest='model_size',
        help='Options:  "full" or "tiny"'
    )

    args = parser.parse_args()
    main(args)