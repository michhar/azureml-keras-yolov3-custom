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
        ws = Workspace.from_config(path='./project/config.json')
    # Need to create the workspace
    except Exception as err:
        print('No workspace.  Check for config.json file under "project/.azureml".')
        assert False

    model = Model.register(model_path=args.model_local,
                        model_name=args.model_workspace,
                        tags={'framework': "keras", 'type': "object detection"},
                        description=args.description,
                        workspace=ws)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-local', type=str, dest='model_local',
        help='The name of the model locally'
    )
    parser.add_argument(
        '--model-workspace', type=str, dest='model_workspace',
        help='The name of the model to be used to register with Azure ML (the "model path" or name in workspace)'
    )
    parser.add_argument(
        '--description', type=str, dest='description', default='Object detection model trained with Keras',
        help='A text description of the model, use and/or data source for reference later.  Shows up in the workspace.'
    )

    args = parser.parse_args()
    main(args)