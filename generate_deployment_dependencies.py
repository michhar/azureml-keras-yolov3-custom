"""
Download the Dockerfile and dependency files to build locally or on device.

https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where#generate-a-dockerfile-and-dependencies
"""
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.model import InferenceConfig

import argparse


def main(args):

    # Define workspace object
    try:    
        ws = Workspace.from_config(path='deploy/.azureml/config.json')
    # Need to create the workspace and download config.json from Azure Portal
    except Exception as err:
        print('No workspace.  Check for deploy/.azureml/config.json file.')
        assert False

    model = Model(ws, name=args.model_workspace)

    inference_config = InferenceConfig(runtime= "python", 
                                entry_script="score.py",
                                conda_file="keras_env.yml",
                                source_directory="./deploy")

    package = Model.package(ws, [model], inference_config, generate_dockerfile=True)
    package.wait_for_creation(show_output=True)
    # Download the package.
    package.save("./" + args.out_dir)
    # Get the Azure container registry that the model/Dockerfile uses.
    acr = package.get_container_registry()
    print("Address:", acr.address)
    print("Username:", acr.username)
    print("Password:", acr.password)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--out-dir', type=str, dest='out_dir',
        help='The name of the local output directory for the Dockerfile and dependencies.'
    )
    parser.add_argument(
        '--model-workspace', type=str, dest='model_workspace',
        help='The name of the model registered with Azure ML (the name in Workspace, not local)'
    )

    args = parser.parse_args()
    main(args)