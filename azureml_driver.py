"""
Azure ML Service driver script for keras experiments.
"""

import argparse
import shutil
import os
import json
import time
import logging

import azureml
from azureml.core import Experiment, Workspace, Run, Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.dataset import Dataset
from azureml.train.dnn import TensorFlow


def main(args):
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
    cluster_name = "gpuforkeras"

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
    experiment_name = args.experiment_name
    experiment = Experiment(ws, name=experiment_name)

    # # Use an AML Data Store for training data
    ds = Datastore.register_azure_blob_container(workspace=ws, 
        datastore_name=args.datastore_name, 
        container_name=os.getenv('STORAGE_CONTAINER_NAME_TRAINDATA', ''),
        account_name=os.getenv('STORAGE_ACCOUNT_NAME', ''), 
        account_key=os.getenv('STORAGE_ACCOUNT_KEY', ''),
        create_if_not_exists=True)

    # Set up for training
    script_params = {
        # --data_path is a Python object that will mount the 
        #   datastore to the compute target in next step (linking 
        #   to Blob Storage)
        '--data_path': ds.as_mount(),
        '--data_dir': args.data_dir,
        '--gpu_num': args.gpu_num,
        '--class_path': args.class_path,
        '--num_clusters': args.num_clusters,
        '--batch_size': args.batch_size,
        '--learning_rate': args.learning_rate
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
                                      'Pillow',
                                      'numpy',
                                      'configparser',
                                      'python-dotenv',
                                      'tensorflow==1.13.1'],
                        use_gpu=True,
                        framework_version='1.13')

    # Submit and wait for run to complete - check experiment in Azure Portal for progress
    run = experiment.submit(estimator)
    print(run.get_details())
    run.wait_for_completion(show_output=True)

    # Register models to Workspace
    model = run.register_model(model_name='keras-dnn-intermediate', 
                               model_path='./outputs/trained_weights_intermediate.h5',
                               tags={'framework': "Keras", 'task': "object detection"},
                               description="Custom Keras YOLOv3 model - before fine-tuning phase")
    model = run.register_model(model_name='keras-dnn', 
                               model_path='./outputs/trained_weights_final.h5',
                               tags={'framework': "Keras", 'task': "object detection"},
                               description="Custom Keras YOLOv3 model - final, after fine-tuning phase")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # Command line options
    parser.add_argument(
        '--experiment-name', type=str, dest='experiment_name',
        help='A name for Azure ML experiment.'
    )

    parser.add_argument(
        '--gpu-num', type=int, dest='gpu_num', default=1,
        help='Number of GPU to use; default is 1.'
    )

    parser.add_argument(
        '--class-path', type=str, dest='class_path',
        help='Text file with class names one per line.'
    )

    parser.add_argument(
        '--data-dir', type=str, dest='data_dir', default='Project-PascalVOC-export',
        help='Directory for training data as found in the Blob Storage container.'
    )

    parser.add_argument(
        '--num-clusters', type=str, dest='num_clusters', default=9,
        help='Number of anchor boxes; 9 for full size YOLO and 6 for tiny YOLO; default is 9.'
    )

    parser.add_argument(
        '--ds-name', type=str, dest='datastore_name',
        help='Name of the Azure ML datastore.'
    )

    parser.add_argument(
        '--bs', type=str, dest='batch_size', default=4,
        help='Batch size (minibatch size for training).'
    )

    parser.add_argument(
        '--lr', type=str, dest='learning_rate', default=0.001,
        help='Learning rate.'
    )

    args = parser.parse_args()

    main(args)