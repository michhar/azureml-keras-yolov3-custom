"""
Azure ML Service driver script for keras experiments.
"""

import azureml
from azureml.core import Experiment
from azureml.core import Workspace, Run, Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.dataset import Dataset
from azureml.train.dnn import TensorFlow

import shutil
import os
import json
import time
import logging


def main():
    logging.info('Main started.')

    # Define workspace object
    try:    
        ws = Workspace.from_config(path='./project/.azureml/config.json')
    # Need to create the workspace
    except Exception as err:
        print('No workspace.  Check for config.json file.')
        assert False
        # ws = Workspace.create(name=os.getenv('WORKSPACE_NAME', ''),
        #             subscription_id=os.getenv('AZURE_SUB', ''), 
        #             resource_group=os.getenv('RESOURCE_GROUP', ''),
        #             create_resource_group=True,
        #             location='westus2'))
        # print("Created workspace {} at location {}".format(ws.name, ws.location))  

    # choose a name for your cluster - under 16 characters
    cluster_name = "gpuforkeras2"

    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing compute target.')
    except ComputeTargetException:
        print('Creating a new compute target...')
        # AML Compute config - if max_nodes are set, it becomes persistent storage that scales
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                            min_nodes=0,
                                                            max_nodes=5)
        # create the cluster
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    # use get_status() to get a detailed status for the current cluster. 
    # print(compute_target.get_status().serialize())

    # # Create a project directory and copy training script to ii
    project_folder = os.path.join(os.getcwd(), 'project')

    # Create an experiment
    experiment_name = 'keras-big-objects'
    experiment = Experiment(ws, name=experiment_name)

    # # Use an AML Data Store for training data
    ds = Datastore.register_azure_blob_container(workspace=ws, 
        datastore_name='azuremlinput3', 
        container_name=os.getenv('STORAGE_CONTAINER_NAME_TRAINDATA', ''),
        account_name=os.getenv('STORAGE_ACCOUNT_NAME', ''), 
        account_key=os.getenv('STORAGE_ACCOUNT_KEY', ''),
        create_if_not_exists=True)

    # Set up for training
    script_params = {
        # --data_dir is a Python object that will mount the 
        #   datastore to the compute target in next step (linking 
        #   to Blob Storage)
        '--data_dir': ds.as_mount(),
        '--model': 'yolov3-custom-base4truck.h5',
        '--gpu_num': 1,
        '--annot_path': 'train_truck.txt',
        '--class_path': 'truck_custom_classes.txt',
        '--anchors_path': 'truck_custom_anchors.txt'
    }

    # Instantiate PyTorch estimator with upload of final model to
    # a specified blob storage container (this can be anything)
    estimator = TensorFlow(source_directory=project_folder,
                        script_params=script_params,
                        compute_target=compute_target,
                        entry_script='train_azureml.py',
                        pip_packages=['keras==2.2.4',
                                      'matplotlib==3.1.1',
                                      'opencv-python==4.1.1.26', 
                                      'Pillow'],
                        use_gpu=True)

    run = experiment.submit(estimator)
    print(run.get_details())
    run.wait_for_completion(show_output=True)

    model = run.register_model(model_name='keras-dnn-truck-intermediate', model_path='./outputs/trained_weights_stage_1.h5')
    model = run.register_model(model_name='keras-dnn-truck', model_path='./outputs/trained_weights_final.h5')
    model = run.register_model(model_name='keras-dnn-truck-intermediate-arch', model_path='./outputs/trained_weights_stage_1.json')
    model = run.register_model(model_name='keras-dnn-truck-arch', model_path='./outputs/trained_weights_final.json')

if __name__ == '__main__':
    main()