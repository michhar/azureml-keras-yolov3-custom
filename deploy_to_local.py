"""
Deploy script for YOLOv3 Keras locally for debugging.  This script
does utilize a cloud registered model (a model registered to the
Azure ML Workspace).

You can use "register_local_model_custom.py" to register the model
if not already done so.
"""
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.webservice import LocalWebservice
from azureml.core.model import Model
import argparse

def main(args):

    # Define workspace object
    try:    
        ws = Workspace.from_config(path='deploy/.azureml/config.json')
    # Need to create the workspace
    except Exception as err:
        print('No workspace.  Check for deploy/.azureml/config.json file.')
        assert False

    inference_config = InferenceConfig(runtime= "python", 
                                    entry_script="score.py",
                                    conda_file="keras_env.yml",
                                    source_directory="./deploy")


    deployment_config = LocalWebservice.deploy_configuration()

    model = Model(ws, name=args.model_workspace)

    # This deploys AND registers model (if not registered)
    service = Model.deploy(workspace=ws, 
                            name=args.service_name, 
                            models=[model], 
                            inference_config=inference_config, 
                            deployment_config=deployment_config)

    service.wait_for_deployment(True)
    print(service.state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--service-name', type=str, dest='service_name',
        help='The name of the model locally'
    )
    parser.add_argument(
        '--model-workspace', type=str, dest='model_workspace',
        help='The name of the model registered with Azure ML (the name in Workspace, not local)'
    )

    args = parser.parse_args()
    main(args)