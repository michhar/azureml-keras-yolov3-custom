"""
Register a local model to Azure ML workspace
"""

import azureml
from azureml.core import Workspace
from azureml.core.model import Model

import os


def main():

    # Use service principal secrets to create authentication vehicle and 
    # define workspace object
    try:    
        ws = Workspace.from_config(path='./project/.azureml/config.json')
    # Need to create the workspace
    except Exception as err:
        print('No workspace.  Check for config.json file.')
        assert False

    # model = Model.register(model_path = "project/yolo.h5",
    #                     model_name = "yolov3.h5",
    #                     tags = {'framework': "keras", 'type': "object detection"},
    #                     description = "Keras base model trained on COCO (80 classes)",
    #                     workspace = ws)

    model = Model.register(model_path="logs/trained_weights_final_ep275.h5",
                        model_name="mixdata_trained_weights.h5",
                        tags={'framework': "keras", 'type': "object detection", "format": "h5", "notes": ""},
                        description="Keras YOLOv3 full-sized model trained - 1 class",
                        workspace=ws)

    model_json = Model.register(model_path="logs/trained_weights_final_ep275.json",
                        model_name="mixdata_trained_weights.json",
                        tags={'framework': "keras", 'type': "object detection", "format": "json", "notes": ""},
                        description="Keras YOLOv3 full-sized model json/architecture trained - 1 class",
                        workspace=ws)

    model_root = Model.get_model_path('logs/')
    print(model_root)


if __name__ == '__main__':
    main()