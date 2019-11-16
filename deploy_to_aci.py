"""
Deploy script for YOLOv3 Keras cloud service to Azure Container Instance
"""
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.webservice import Webservice
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

    aciconfig = AciWebservice.deploy_configuration(cpu_cores=1,
                                               auth_enabled=True, # this flag generates API keys to secure access
                                               memory_gb=6,
                                               location="westus",
                                               tags={'name': 'yolov3_full', 'framework': 'Keras'},
                                               description='Keras YOLOv3 full size for object detection')

    model = Model(ws, name='mixdata_trained_weights.h5')

    # This deploys AND registers model (if not registered)
    service = Model.deploy(workspace=ws, 
                            name='keras-yolov3-service', 
                            models=[model], 
                            inference_config=inference_config, 
                            deployment_config=aciconfig)

    # This just deploys and does not register
    # service = Webservice.deploy_from_model(ws, 
    #                             name='keras-yolov3-service', 
    #                             models=[model], 
    #                             deployment_config=aciconfig)

    service.wait_for_deployment(True)
    print(service.state)

if __name__ == "__main__":
    main()