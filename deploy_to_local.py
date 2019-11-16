"""
Deploy script for YOLOv3 Keras cloud service to Azure Container Instance
"""
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.webservice import LocalWebservice
from azureml.core.model import Model

def main():

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

    model = Model(ws, name='mixdata_trained_weights.h5')

    # This deploys AND registers model (if not registered)
    service = Model.deploy(workspace=ws, 
                            name='keras-yolov3-service-2', 
                            models=[model], 
                            inference_config=inference_config, 
                            deployment_config=deployment_config)

    service.wait_for_deployment(True)
    print(service.state)

if __name__ == "__main__":
    main()